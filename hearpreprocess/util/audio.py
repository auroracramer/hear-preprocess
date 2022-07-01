"""
Audio utility functions for evaluation task preparation
"""

import json
import random
import os
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import luigi
import numpy as np
import soundfile as sf
import ffmpeg
import pedalboard
import pedalboard.io
from tqdm import tqdm
from pedalboard import VST3Plugin

from hearpreprocess.util.misc import opt_dict
from hearpreprocess.util.luigi import WorkTask


class VST3PluginTask(WorkTask):
    vst_path = luigi.Parameter()
    _vst: Optional[VST3Plugin] = None

    @staticmethod
    def set_vst_params(vst):
        raise NotImplementedError("Deriving classes need to implement this")

    @staticmethod
    def load_vst(vst_path):
        assert vst_path is not None, "Required VST path was not provided"
        assert os.path.exists(vst_path), "Cannot find VST path"
        vst = pedalboard.load_plugin(vst_path)
        self.set_vst_params(vst)
        return vst

    def process(self, inp_path, out_path):
        with pedalboard.io.AudioFile(os.path.realpath(inp_path), 'r') as f:
            inp_audio = f.read(f.frames)
            sr = f.samplerate
        out_audio = self._vst(inp_audio, sr)
        sf.write(out_path, out_audio, sr)
        
    @property
    def _vst(self):
        if self._vst is None:
            vst_path =  self.task_config["vst3_paths"]["IEM/BinauralDecoder"]
            self._vst = self.load_vst(vst_path)
        return self._metadata


class FOAToBinauralTask(VST3PluginTask):
    @staticmethod
    def set_vst_params(vst):
        vst.bypass = False
        vst.input_ambisonic_order = "1st"
        vst.input_normalization = "N3D"
    

def to_wav(in_file: str, out_file: str,
            exp_in_chan: Optional[int] = None, out_chan: Optional[int] = None,
            max_chan: Optional[int] = None) -> None:
    """converts the audio to wav format"""
    assert not Path(out_file).exists(), "File already exists"
    in_stats = get_audio_stats(in_file)
    assert (
        not max_chan
        or (in_stats["channels"] <= max_chan
        and (not out_chan or out_chan <= max_chan))
        and (not exp_in_chan or exp_in_chan <= max_chan)
    ), f"Only up to {max_chan} channels are valid"
    assert (
        not exp_in_chan or in_stats["channels"] == exp_in_chan
    ), f"Expected {exp_in_chan} input channels, got {in_stats['channels']}"
    if not (in_stats and in_stats["codec"] == "pcm_s16le" and (
                not out_chan or in_stats["channels"] == out_chan)):
        try:
            # By default keep the same number of channels
            _ = (
                ffmpeg.input(in_file)
                .audio.output(out_file, f="wav", acodec="pcm_s16le",
                                **opt_dict("ac", out_chan, bool(out_chan)))
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(
                "Please check the console output for ffmpeg to debug the "
                "error in mono wav: ",
                f"Error: {e}",
            )
            raise
        # Check if the generated file is present and that ffmpeg can
        # read stats for the file to be used in subsequent processing steps
        assert Path(out_file).exists(), "wav file saved by ffmpeg was not found"
        out_stats = get_audio_stats(out_file)
        assert (
            out_stats
            and out_stats["ext"] == ".wav"
            and out_stats["codec"] == "pcm_s16le"
            and ((not out_chan) or (out_stats["channels"] ==  out_chan))
        ), "Unable to get stats for the generated wav file"
    else:
        Path(out_file).symlink_to(Path(in_file).absolute())


def naive_mono_wav(in_file: str, out_file: str) -> None:
    """converts the audio to wav format with mono stream"""
    return to_wav(in_file, out_file, out_chan=1, max_chan=2)


def channel_reformat_wav(
    in_file: str, out_file: str, in_chfmt: str, out_chfmt: str, vst3_task: Optional[VST3PluginTask] = None) -> None:
    """converts the audio to the target channel format"""

    assert in_chfmt in ("foa", "mono", "stereo", "mixed_mono_stereo"), f"Invalid channel format {in_chfmt}"
    assert out_chfmt in ("foa", "mono", "stereo"), f"Invalid channel format {out_chfmt}"

    if out_chfmt == "foa":
        assert (in_chfmt not in ("mono", "mixed_mono_stereo", "stereo")
        ), "Upmixing from mono or stereo to FOA is unsupported"
        # in_chfmt == "foa"
        return to_wav(in_file, out_file, exp_in_chan=4, out_chan=4)
    elif out_chfmt == "stereo":
        assert (in_chfmt not in ("mono", "mixed_mono_stereo")
        ), "Upmixing from mono to stereo is unsupported"
        if in_chfmt == "foa":
            return foa_to_stereo_wav(in_file, out_file, vst3_task=vst3_task)
        else: # in_chfmt == "stereo"
            return to_wav(in_file, out_file, exp_in_chan=2, out_chan=2)
    else: # out_chfmt == "mono"
        if in_chfmt == "foa":
            return foa_to_mono_wav(in_file, out_file)
        elif in_chfmt == "stereo":
            return to_wav(in_file, out_file, exp_in_chan=2, out_chan=1)
        elif in_chfmt == "mixed_mono_stereo":
            return to_wav(in_file, out_file, max_chan=2, out_chan=1)
        else: # mono
            return to_wav(in_file, out_file, exp_in_chan=1, out_chan=1)



def foa_to_stereo_wav(in_file: str, out_file: str, vst3_task: Optional[VST3PluginTask] = None) -> None:
    """converts the FOA audio to wav format with stereo stream"""
    vst3_task.process(in_file, out_file)


def foa_to_mono_wav(in_file: str, out_file: str) -> None:
    """converts the FOA audio to wav format with mono stream"""
    assert not Path(out_file).exists(), "File already exists"
    in_stats = get_audio_stats(in_file)
    assert not (in_stats and in_stats["codec"] == "pcm_s16le" 
                and in_stats["channels"] == 1), "Expected FOA but got mono"
    assert in_stats["channels"] == 4, "Must have 4 channels for FOA audio"
    try:
        # Extract the W channel for mono
        _ = (
            ffmpeg.input(in_file)
            .output(out_file, f="wav", acodec="pcm_s16le",
                    ac=1, af="pan=mono|c0=c0")
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print(
            "Please check the console output for ffmpeg to debug the "
            "error in FOA wav: ",
            f"Error: {e}",
        )
        raise
    # Check if the generated file is present and that ffmpeg can
    # read stats for the file to be used in subsequent processing steps
    assert Path(out_file).exists(), "wav file saved by ffmpeg was not found"
    out_stats = get_audio_stats(out_file)
    assert (
        out_stats
        and out_stats["ext"] == ".wav"
        and out_stats["codec"] == "pcm_s16le"
        and out_stats["channels"] == 4
    ), "Unable to get stats for the generated wav file"


def trim_pad_wav(in_file: str, out_file: str, duration: float) -> None:
    """
    Trims and pads the audio to the desired output duration
    If the audio is already of the desired duration, make a symlink
    """
    assert not Path(out_file).exists(), "File already exists"
    in_stats = get_audio_stats(in_file)
    assert in_stats["codec"] == "pcm_s16le"
    # If the audio is of the desired duration
    # move to the else part where we will just create a symlink
    if in_stats["duration"] != duration:
        # Trim and pad the audio
        try:
            _ = (
                ffmpeg.input(in_file)
                .audio.filter("apad", whole_dur=duration)  # Pad
                .filter("atrim", end=duration)  # Trim
                .output(out_file, f="wav", acodec="pcm_s16le") # don't set ac=1 so it works with other sample rates
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(
                "Please check the console output for ffmpeg to debug the "
                "error in trim and pad wav: ",
                f"Error: {e}",
            )
            raise
        # Check if the file has been converted to the desired duration
        out_stats = get_audio_stats(out_file)
        assert (
            out_stats["duration"] == duration
        ), f"The new file is {out_stats['duration']} secs "
        f"while expected is {duration} secs"
        assert out_stats["codec"] == "pcm_s16le"
    else:
        Path(out_file).symlink_to(Path(in_file).absolute())


def resample_wav(in_file: str, out_file: str, out_sr: int) -> None:
    """
    Resample a wave file using SoX high quality mode
    If the audio is already of the desired sample rate, make a symlink
    """
    assert not Path(out_file).exists()
    in_stats = get_audio_stats(in_file)
    assert in_stats["codec"] == "pcm_s16le"
    # If the audio is of the desired sample rate
    # move to the else part where we will just create a symlink
    if in_stats["sample_rate"] != out_sr:
        try:
            _ = (
                ffmpeg.input(in_file)
                # Use SoX high quality mode
                .filter("aresample", resampler="soxr")
                .output(out_file, ar=out_sr)
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(
                "Please check the console output for ffmpeg to debug the "
                "error in resample wav: ",
                f"Error: {e}",
            )
            raise
        # Check if the file has been converted to the desired sampling rate
        out_stats = get_audio_stats(out_file)
        assert (
            out_stats["sample_rate"] == out_sr
        ), f"The new file is {out_stats['sample_rate']} secs "
        f"while expected is {out_sr} secs"
        assert out_stats["codec"] == "pcm_s16le"
    else:
        # If the audio has the expected sampling rate, make a symlink
        Path(out_file).symlink_to(Path(in_file).absolute())


def get_audio_stats(in_file: Union[str, Path]) -> Union[Dict[str, Any], Any]:
    """Produces summary for a single audio file"""
    try:
        audio_stream = ffmpeg.probe(in_file, select_streams="a")["streams"][0]
        audio_stats = {
            "codec": audio_stream["codec_name"],
            "sample_rate": int(audio_stream["sample_rate"]),
            "samples": int(audio_stream["duration_ts"]),
            "mono": audio_stream["channels"] == 1,
            "stereo": audio_stream["channels"] == 2,
            "foa": audio_stream["channels"] == 4,
            "channels": audio_stream["channels"],
            "duration": float(audio_stream["duration"]),
            "ext": Path(in_file).suffix,
        }
    except (ffmpeg.Error, KeyError):
        # Skipping audio file for stats calculation.
        return None
    return audio_stats


def get_audio_dir_stats(
    in_dir: Union[str, Path], out_file: str, exts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Produce summary by recursively searching a directory for wav files"""
    MAX = 1000

    if exts is None:
        exts = [".wav", ".mp3", ".ogg", ".webm"]

    # Get all the audio files with glob. Path.rglob doesnot recursively
    # look into symlinked folders. Using glob with recursive True, looks inside
    # symlinked folders as well
    all_file_paths = map(Path, glob(str(Path(in_dir).joinpath("**/*")), recursive=True))
    audio_paths = list(
        filter(
            lambda audio_path: audio_path.suffix.lower()
            in map(str.lower, exts),  # type: ignore
            all_file_paths,
        )
    )
    if len(audio_paths) == 0:
        print("No audio files present in the folder")
        return {}
    rng = random.Random(0)
    rng.shuffle(audio_paths)

    orig_count = len(audio_paths)
    audio_paths = audio_paths[:MAX]

    # Count the number of successful and failed statistics extraction to be
    # added the output stats file
    success_counter: Dict[str, int] = defaultdict(int)
    failure_counter: Dict[str, int] = defaultdict(int)

    # Iterate and get the statistics for each audio
    audio_dir_stats = []
    for audio_path in tqdm(audio_paths):
        audio_stats = get_audio_stats(audio_path)
        if audio_stats is not None:
            audio_dir_stats.append(audio_stats)
            success_counter[audio_path.suffix] += 1
        else:
            # update the failed counter if the extraction was not
            # succesful
            failure_counter[audio_path.suffix] += 1

    assert audio_dir_stats, "Stats was not calculated for any audio file. Please Check"
    " the formats of the audio file"
    durations = [stats["duration"] for stats in audio_dir_stats]
    unique_sample_rates = dict(
        Counter([stats["sample_rate"] for stats in audio_dir_stats])
    )
    unique_num_channels = dict(
        Counter([stats["channels"] for stats in audio_dir_stats])
    )
    mono_audio_count = sum(stats["mono"] for stats in audio_dir_stats)
    stereo_audio_count = sum(stats["stereo"] for stats in audio_dir_stats)
    foa_audio_count = sum(stats["foa"] for stats in audio_dir_stats)

    summary_stats: Dict[str, Any] = {"count": orig_count}
    if len(audio_paths) != orig_count:
        summary_stats.update({"count_sample": len(audio_paths)})

    duration = {
        "mean": round(np.mean(durations), 2),
        "var": round(np.var(durations), 2),
    }
    if np.var(durations) > 0.0:
        duration.update(
            {
                "min": round(np.min(durations), 2),
                "max": round(np.max(durations), 2),
                # Percentile duration of the audio
                **{
                    f"{p}th": round(np.percentile(durations, p), 2)
                    for p in [10, 25, 50, 75, 90, 95]
                },
            }
        )
    summary_stats.update(
        {
            "duration": duration,
            "samplerates": unique_sample_rates,
            "numchannels": unique_num_channels,
            "count_mono": mono_audio_count,
            "count_stereo": stereo_audio_count,
            "count_foa": foa_audio_count,
            # Count of no of success and failure for audio summary extraction for each
            # extension type
            "summary": {
                "successfully_extracted": dict(success_counter),
                "failed_to_extract": dict(failure_counter),
            },
        }
    )

    json.dump(summary_stats, open(out_file, "w"), indent=True)
    return summary_stats
