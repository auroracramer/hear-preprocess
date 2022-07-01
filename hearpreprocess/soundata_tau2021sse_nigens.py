#!/usr/bin/env python3
import logging
import luigi
import pandas as pd
from multiprocessing.sharedctypes import Value
from typing import Any, Dict

import hearpreprocess.soundata_pipeline as soundata_pipeline
import hearpreprocess.util.units as units_utils
from hearpreprocess.util.misc import opt_list, opt_tuple
from hearpreprocess.pipeline import (
    TEST_PERCENTAGE,
    TRAIN_PERCENTAGE,
    TRAINVAL_PERCENTAGE,
    VALIDATION_PERCENTAGE,
)

logger = logging.getLogger("luigi-interface")

generic_task_config = {
    # Define the task name and the version
    "task_name": "tfds_speech_commands",
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
    "soundata_audio_remotes": [
        {
            "split": "train",
            "remote": "foa_dev"
        },
        {
            "split": "test",
            "remote": "foa_eval"
        },
    ],
    "soundata_meta_remotes": [
        {
            "name": "metadata_train",
            "remote": "metadata_dev"
        },
        {
            "name": "metadata_eval",
            "remote": "metadata_eval"
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

class ExtractMetadata(soundata_pipeline.ExtractSpatialEventsMetadata):
    metadata_train = luigi.TaskParameter()
    metadata_eval = luigi.TaskParameter()

    @staticmethod
    def skip_clip_id(clip_id, split) -> bool:
        # the TAU 2021 SSE NIGENS class doesn't have a split property,
        # so we infer it from the clip_id
        fmt = self.task_config["in_channel_format"]
        if fmt == "foa":
            prefix = "foa"
        else:
            raise ValueError(f"Unsupported audio format {fmt}")

        if split == "train":
            suffix = "dev"
        elif split == "test":
            suffix = "eval"
        else:
            raise ValueError(f"Invalid split: {split}")

        start = f"{prefix}_{suffix}"

        return not clip_id.startswith(start)
        

    def requires(self):
        return {"train": self.train, "test": self.test}




def extract_metadata_task(task_config: Dict[str, Any]) -> ExtractMetadata:
    # Build the dataset pipeline with the custom metadata configuration task.
    # Please note the tfds download and extract tasks are used to download and the
    # extract the tensorflow data splits below
    download_tasks = soundata_pipeline.get_download_and_extract_tasks_soundata(task_config)
    
    # Set up constructor based on annotation type
    annotation_type = task_config["soundata_annotation_type"]
    return ExtractMetadata(
        outfile="process_metadata.csv", task_config=task_config, **download_tasks
    )