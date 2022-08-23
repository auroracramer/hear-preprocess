#!/usr/bin/env python3
from __future__ import annotations
import logging
import os
import luigi
import numpy as np
import pandas as pd
from functools import partial
from pathlib import Path
from statistics import mode
from typing import Any, Dict, Set
from urllib.parse import urlparse
from slugify import slugify

import hearpreprocess.pipeline as pipeline
import hearpreprocess.util.units as units_util
from hearpreprocess.pipeline import (
    TRAIN_PERCENTAGE,
    TRAINVAL_PERCENTAGE,
    VALIDATION_PERCENTAGE,
    DownloadCorpus,
    ExtractArchive,
)
from hearpreprocess.util.misc import opt_list, opt_tuple

logger = logging.getLogger("luigi-interface")

generic_task_config = {
    # Define the task name and the version
    "task_name": "urbansas_pointwise",
    "version": "hear2021-ext",
    "embedding_type": "event", # we'll only support "event" for seld
    "prediction_type": "avoseld_multiregion",
    "split_mode": "new_split_stratified_kfold",
    "nfolds": 5,
    "stratify_fields": ["location_id"],
    "sample_duration": 10.0,
    "evaluation": ["horiz_iou_120fov_5regions_pointwise"],
    "spatial_projection": "video_azimuth_region_pointwise",
    # This task uses tfds which doesn't require the download paths,
    # but rather the tfds dataset name and version. For speech commands
    # the tf dataset has all the splits, and so we will not be
    # doing any custom splitting. All the splits will be extracted
    # from tfds builder.
    "in_channel_format": "stereo", # ("mono_stereo", "micarray", "foa")
    "fov": 120.0,
    "frame_width": 1280, # NEED TO CHECK THISSSS
    "index_max_box_only": False,
    "num_regions": 5,
    "filter_confirmed": True, # bool or float in [0, 1]
    "download_urls": [
        {
            "name": "audio",
            "url": "https://zenodo.org/record/6658386/files/audio.zip?download=1",  # noqa: E501
            "md5": "a870dcd7791df99c8a19ed6da864f8ca",
        },
        {
            "name": "annotations",
            "url": "https://zenodo.org/record/6658386/files/annotations.zip?download=1",  # noqa: E501
            "md5": "02a1ce5a50857aa46dfceb2f43279c76",
        },
    ],
    "default_mode": "5h",
    "modes": {
        # Different modes for different max task duration
        "5h": {
            "max_task_duration_by_fold": 3600 * 5,
        },
        "full": {
            "max_task_duration_by_fold": None,
        },
    },
    "evaluation_params": {
        "event_postprocessing_grid": {
            # In preliminary tests, these smoothing parameters worked
            # well at optimizing onset fms.
            "threshold_multitrack_unify": [30.0],
        },
    },
}

def pos2angle(x: np.ndarray, fov: float, frame_width: float) -> np.ndarray:
    """
    Given a position in reference to an image coordinate, convert it to an angle.
    
    Arguments:
        x: position in reference to an image coordinate.
        fov: field of video.
        frame_width: width of video frame.
    
    Returns:
        azimuth angle mapped from the given position.
    """
    # Largely copied from: https://github.com/magdalenafuentes/urbansas/blob/main/index/audio_visual_dataset.py
    return (x / frame_width - 1 / 2) * fov


def filter_for_largest_box(track_df: pd.DataFrame, top_n: int=1) -> pd.DataFrame:
    """
    Filter the video annotations for one file down to only the largest bounding
    box annotation for each class per frame.
    Arguments:
        track_df: video annotations for one video.
        top_n: top "N" largest bounding boxes to filter to.
    Returns:
        pd.DataFrame: video annotations filtered to the top N largest 
            bounding box annotations per class and frame.
    """
    # Largely copied from: https://github.com/magdalenafuentes/urbansas/blob/main/index/audio_visual_dataset.py

    trimmed_video_annotations = pd.DataFrame(columns=track_df.columns)

    for t in np.arange(0, 10, 0.5):
        for c in list(set(track_df['label'])):
            curr_records = track_df.loc[((track_df['label'] == c) & (track_df['time'] == t))]
            box_areas = []
            if len(curr_records) > 0:
                for i, r in curr_records.iterrows():
                    box_areas.append(r['w'] * r['h'])
                curr_records['box_area'] = box_areas
                sorted_records = curr_records.sort_values(by='box_area', ascending=False)
                top_records = sorted_records.head(top_n)
                top_records = top_records.drop(columns='box_area')
                trimmed_video_annotations = trimmed_video_annotations.append(top_records, ignore_index=True) 
    return 


def process_video_track(track_df, fov=None, frame_width=None) -> pd.Series:
    """
    Processes video annotations data by cleaning up typing and
    converting bounding box positions to azimuth angles.
    Arguments: 
        track_df: video annotations for one video.
        fov: field of view.
        frame_width: width of video frame.
    Returns: 
        pd.Series: processed video track.
    """
    
    x = track_df.x.values
    w = track_df.w.values

    track_ds = pd.Series({
        'label': mode(track_df.label),
        'time': track_df.time.to_list(),
        'azimuth': list(pos2angle(x + w/2, fov=fov, frame_width=frame_width)),
        'azimuth_left': list(pos2angle(x, fov=fov, frame_width=frame_width)),
        'azimuth_right': list(pos2angle(x + w, fov=fov, frame_width=frame_width)),
        'visibility': track_df.visibility.to_list(),
    })
    return track_ds


def is_number(x) -> bool:
    return isinstance(x, (float, int)) and not isinstance(x, bool)


class ExtractMetadata(pipeline.ExtractMetadata):
    audio = luigi.TaskParameter()
    annotations = luigi.TaskParameter()

    def requires(self):
        return {
            "audio": self.audio,
            "annotations": self.annotations,
        }

    def get_requires_metadata(self, requires_key: str) -> pd.DataFrame:
        assert requires_key == "audio", f"split must be 'audio', but got {requires_key}"
        logger.info(f"Preparing metadata")
        # Largely copied from: https://github.com/magdalenafuentes/urbansas/blob/main/index/audio_visual_dataset.py

        audio_dir = self.requires()["audio"].output_path.joinpath("audio")
        annotations_dir = self.requires()["annotations"].output_path.joinpath("annotations")
        audio_metadata_path = annotations_dir.joinpath("audio_annotations.csv")
        video_metadata_path = annotations_dir.joinpath("video_annotations.csv")

        fov = self.task_config["fov"]
        frame_width = self.task_config["frame_width"]
        index_max_box_only = self.task_config["index_max_box_only"]
        prediction_type = self.task_config["prediction_type"]
        filter_confirmed = self.task_config["filter_confirmed"]
        assert prediction_type in ("avoseld_multiregion", "multilabel"),(
            f"prediction_type must be in "
            f"('avoseld_multiregion', 'multilabel', "
            f"but got {prediction_type}"
        )


        df_audio = pd.read_csv(audio_metadata_path)
        df_video = pd.read_csv(video_metadata_path)

        # Compile filenames list
        filenames_list = sorted(set(df_audio['filename'].to_list() +
                                    df_video['filename'].to_list()))

        metadatas = []
        # Iterate on metadata and files
        for filename in filenames_list:

            audio_files = list(Path(audio_dir).glob(filename + '.*'))
            if len(audio_files) == 0:
                raise FileExistsError(
                    f'Audio file not found: {filename}\n'
                    f'Audio folder: {audio_dir}\n'
                    f'Please download the full dataset first'
                )
            if len(audio_files) > 1:
                raise FileExistsError(
                    f'Multiple audio files found for: {filename}\n'
                    f'Audio folder: {audio_dir}\n'
                    f'Please clean your dataset folder first'
                    )
            audio_path = audio_files[0]

            df_audio_file = df_audio[df_audio['filename'] == filename]

            if prediction_type == "multilabel":
                # Plain ol' SED
                metadata = df_audio_file[["start", "end", "label"]]

            elif prediction_type == "avoseld_multiregion":
                pointwise = self.task_config["spatial_projection"] == "video_azimuth_region_pointwise"
                boxwise = self.task_config["spatial_projection"] == "video_azimuth_region_boxwise"
                columns = (
                    ["start", "end", "trackidx", "label"]
                    + opt_list("azimuth", pointwise)
                    + opt_list("azimuthleft", boxwise)
                    + opt_list("azimuthright", boxwise)
                )
                # Audio-visual object localization and detection
                df_video_file = df_video[df_video['filename'] == filename]

                if len(df_video_file):
                    city = df_video_file['city'].iloc[0]
                    location_id = df_video_file['location_id'].iloc[0]
                else:
                    stem = Path(df_audio_file.iloc[0]['filename']).stem
                    city = stem.split('-')[1] if stem.startswith('street-traffic') else 'montevideo'
                    location_id = stem.split('_')[0][:-4] if city == 'montevideo' else '-'.join(stem.split('-')[2:4])

                if index_max_box_only:
                    df_video_file = filter_for_largest_box(df_video_file, top_n=1)

                # Determine objects from video
                df_video_objects = df_video_file.groupby('track_id').apply(
                    partial(process_video_track, fov=fov, frame_width=frame_width)
                )

                # Index audio evdents
                audio_events_list = []
                file_meta = {}
                for _, ods in df_audio_file.iterrows():
                    if str(ods['label']) != "-1":
                        audio_events_list.append({
                            'label': ods['label'],
                            'start': ods['start'],
                            'end': ods['end'],
                        })
                    file_meta.update({
                        'non_identifiable_vehicle_sound': bool(ods['non_identifiable_vehicle_sound'])
                    })

                metadata = []
                for track_id, ods in df_video_objects.iterrows():
                    event_dict = ods.to_dict()

                    label = ods["label"]
                    # Check if video event is confirmed by audio.
                    # This is true if at least half the timestamp from video labels are included in audio regions
                    confirmed_list = [
                        any((t >= aev['start'] and t <= aev['end']) and
                            aev['label'] == event_dict['label']
                            for aev in audio_events_list)
                        for t in event_dict['time']
                    ]

                    # https://github.com/magdalenafuentes/urbansas/blob/main/data/BatchRawDataset.py#L192
                    confirmed_ratio = np.mean(confirmed_list)
                    if is_number(filter_confirmed):
                        if (confirmed_ratio <= filter_confirmed):
                            continue
                    elif isinstance(filter_confirmed, bool):
                        # If bool, confirmation just requires at least one
                        # frame to be audio-confirmed
                        if filter_confirmed and (confirmed_ratio == 0):
                            continue
                    else:
                        raise ValueError(
                            f"filter_confirmed must be numeric or bool, but "
                            f"got {type(filter_confirmed).__name__}"
                        )

                    for idx, confirmed in enumerate(confirmed_list):
                        # Skip unconfirmed
                        if not confirmed:
                            continue
                        
                        time = event_dict["time"][idx]
                        azimuth = event_dict["azimuth"][idx]
                        azimuth_left = event_dict["azimuth_left"][idx]
                        azimuth_right = event_dict["azimuth_right"][idx]

                        # Define start time to be the frame time
                        start = time
                        
                        # Define end time to be the next event-track frame
                        if idx < (len(confirmed_list) - 1):
                            end = event_dict["time"][idx+1]
                        else:
                            # or the same time if this is the last frame
                            end = time

                        t_start, t_end = (
                            units_util.norm_time(t, "seconds")
                            for t in (start, end)
                        )
                        row = (
                            (start, end, track_id, label)
                            + opt_tuple(azimuth, pointwise)
                            + opt_tuple(azimuth_left, boxwise)
                            + opt_tuple(azimuth_right, boxwise)
                        )
                            
                        metadata.append(row)


                metadata = pd.DataFrame(metadata, columns=columns)

            metadata = metadata.assign(
                relpath=str(audio_path),
                split="train",
                location_id=location_id,
                city=city,
            )
            metadatas.append(metadata)

        metadata = pd.concat(metadatas)
        return metadata

    def get_all_metadata(self) -> pd.DataFrame:
        return self.get_requires_metadata_check("audio").reset_index(drop=True)


def get_download_and_extract_tasks(task_config: Dict):
    """
    Iterates over the dowload urls and builds download and extract
    tasks for them

    This is redefined in this task (urbansas), and makes sure the directory
    structure is okay
    """

    tasks = {}
    outdirs: Set[str] = set()
    for urlobj in task_config["download_urls"]:
        url, md5 = urlobj["url"], urlobj["md5"]
        filename = os.path.basename(urlparse(url).path)
        download_task = DownloadCorpus(
            url=url, outfile=filename, expected_md5=md5, task_config=task_config
        )
        outdir = urlobj["name"]

        assert outdir not in outdirs, f"{outdir} in {outdirs}. If you are downloading "
        "multiple archives into one split, they should have different 'name's."

        outdirs.add(outdir)
        task = ExtractArchive(
            download=download_task,
            infile=filename,
            outdir=outdir,
            task_config=task_config,
        )
        tasks[slugify(outdir, separator="_")] = task

    return tasks


def extract_metadata_task(task_config: Dict[str, Any]) -> ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task.
    # Please note the tfds download and extract tasks are used to download and the
    # extract the tensorflow data splits below
    download_tasks = get_download_and_extract_tasks(task_config)

    # Set up constructor based on annotation type
    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )