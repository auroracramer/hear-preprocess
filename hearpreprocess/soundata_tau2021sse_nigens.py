#!/usr/bin/env python3
import logging
import os
import luigi
import pandas as pd
from multiprocessing.sharedctypes import Value
from typing import Any, Dict
from pathlib import Path

import hearpreprocess.soundata_pipeline as soundata_pipeline
from hearpreprocess.util.misc import opt_list, opt_tuple, first
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
    "prediction_type": "seld", # TODO implement prediction type
    "split_mode": "trainvaltest",
    "sample_duration": 1.0,
    "evaluation": ["top1_acc"], # TODO: gotta change this
    # This task uses tfds which doesn't require the download paths,
    # but rather the tfds dataset name and version. For speech commands
    # the tf dataset has all the splits, and so we will not be
    # doing any custom splitting. All the splits will be extracted
    # from tfds builder.
    "in_channel_format": "foa", # ("mono_stereo", "micarray", "foa")
    "soundata_dataset_name": "tau2021sse_nigens",
    # By default all the splits for the above task and version will
    # be downloaded, the below key helps to select the splits to extract
    "soundata_annotation_type": "spatial_events",
    "soundata_valid_spatial_events": {
        "azimuth": True,
        "elevation": True,
        "distance": False
    },
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
                    "prefix_list": ["foa_dev/dev-valid", "foa_dev/dev-test"]
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