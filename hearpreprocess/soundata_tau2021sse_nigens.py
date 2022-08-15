#!/usr/bin/env python3
import logging
import os
import luigi
from typing import Any, Dict

import hearpreprocess.soundata_pipeline as soundata_pipeline
from hearpreprocess.util.misc import first
from hearpreprocess.pipeline import (
    TRAIN_PERCENTAGE,
    TRAINVAL_PERCENTAGE,
    VALIDATION_PERCENTAGE,
)

logger = logging.getLogger("luigi-interface")

generic_task_config = {
    # Define the task name and the version
    "task_name": "soundata_tau2021sse_nigens",
    "version": "hear2021-ext",
    "embedding_type": "event", # we'll only support "event" for seld
    "prediction_type": "seld",
    "multitrack": True,
    "num_tracks": 3,
    "split_mode": "trainvaltest",
    "sample_duration": 60.0,
    "evaluation": ["segment_1s_seld"],
    # This task uses tfds which doesn't require the download paths,
    # but rather the tfds dataset name and version. For speech commands
    # the tf dataset has all the splits, and so we will not be
    # doing any custom splitting. All the splits will be extracted
    # from tfds builder.
    "in_channel_format": "foa", # ("mono_stereo", "micarray", "foa")
    "spatial_projection": "unit_sphere", # none, unit_sphere, unit_xy_disc, unit_yz_disc
    "soundata_dataset_name": "tau2021sse_nigens",
    # By default all the splits for the above task and version will
    # be downloaded, the below key helps to select the splits to extract
    "soundata_annotation_type": "spatial_events",
    "soundata_valid_spatial_events": {
        "azimuth": True,
        "elevation": True,
        "distance": False
    },
    # TAU 2021 SSE NIGENS doesn't include the vocabulary in the dataset files
    "soundata_remap_labels": {
        "0":  "alarm",
        "1":  "crying-baby",
        "2":  "crash",
        "3":  "barking-dog",
        "4":  "female-scream",
        "5":  "female-speech",
        "6":  "footsteps",
        "7":  "knocking-on-door",
        "8":  "male-scream",
        "9":  "male-speech",
        "10": "ringing-phone",
        "11": "piano",
    },
    "soundata_metadata_clip_attrs": ["location_id"], 
    "soundata_splits": [
        {
            "split": "train",
            "remotes": ["foa_dev", "metadata_dev"],
            "filters": [
                {
                    "type": "clip_id_prefix",
                    "prefix_list": ["foa_dev/dev-train"]
                }
            ]
        },
        {
            "split": "valid",
            "remotes": ["foa_dev", "metadata_dev"],
            "filters": [
                {
                    "type": "clip_id_prefix",
                    "prefix_list": ["foa_dev/dev-val", "foa_dev/dev-test"]
                }
            ]
        },
        {
            "split": "test",
            "remotes": ["foa_eval", "metadata_eval"],
            "filters": [
                {
                    "type": "clip_id_prefix",
                    "prefix_list": ["foa_eval/eval-test"]
                }
            ]
        },
    ],
    "default_mode": "5h",
    "modes": {
        # Different modes for different max task duration
        "5h": {
            # No more than 5 hours of audio (training + validation)
            "max_task_duration_by_split": {
                "train": 3600 * 5 * TRAIN_PERCENTAGE / TRAINVAL_PERCENTAGE,
                "valid": 3600 * 5 * VALIDATION_PERCENTAGE / TRAINVAL_PERCENTAGE,
                "test": 3600 * 5,
            }
        },
        "full": {
            "max_task_duration_by_split": {"train": None, "valid": None, "test": None},
        },
    },
}


class DownloadExtractSoundata(soundata_pipeline.DownloadExtractSoundata):
    "Download and extract the Soundata dataset"

    @property
    def output_path(self):
        assert self.task_config["in_channel_format"] == "foa"
        assert len(self.remotes) == 2
        # Get remote name corresponding to audio
        prefix = first(
            remote for remote in self.remotes
            if not remote.startswith("metadata")
        )
        clip_audio_paths = [
            self.dataset.clip(clip_id).audio_path
            for clip_id in self.dataset.clip_ids
            if clip_id.startswith(prefix)
        ]
        if not clip_audio_paths:
            raise ValueError("No valid output path")

        # Return the directory containing all of the audio encompassed
        return os.path.commonpath(clip_audio_paths)


class ExtractMetadata(soundata_pipeline.ExtractSpatialEventsMetadata):
    train = luigi.TaskParameter()
    valid = luigi.TaskParameter()
    test = luigi.TaskParameter()

    def requires(self):
        return {
            "train": self.train,
            "valid": self.valid,
            "test": self.test
        }


def extract_metadata_task(task_config: Dict[str, Any]) -> ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task.
    # Please note the tfds download and extract tasks are used to download and the
    # extract the tensorflow data splits below
    download_tasks = soundata_pipeline.get_download_and_extract_tasks_soundata(
        task_config, DownloadExtractSoundata
    )
    
    # Set up constructor based on annotation type
    annotation_type = task_config["soundata_annotation_type"]
    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )