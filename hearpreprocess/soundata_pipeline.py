#!/usr/bin/env python3
"""
Custom Preprocessing pipeline for tensorflow dataset

soundata audio datasets can be preprocessed with the hear-preprocess pipeline by defining
the generic_task_config dict and optionally overriding the extract metadata in this
file
See example soundata_speech_commands.py for a sample way to configure this for a soundata data

Tasks in this file helps to download and extract the soundata as wav files, followed
by overriding the extract metadata function to consume the extracted audio files and
labels. This is connected to downstream tasks from the main pipeline.

"""
import logging
from multiprocessing.sharedctypes import Value
from pathlib import Path
from decimal import Decimal
from typing import Any, Dict, Optional

import os
import luigi
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import soundata
from slugify import slugify
from tqdm import tqdm

import hearpreprocess.pipeline as pipeline
import hearpreprocess.util.luigi as luigi_util
import hearpreprocess.util.units as units_util

logger = logging.getLogger("luigi-interface")


def get_soundata_dataset(task_config, workdir):
    """
    Returns the soundata object which can be used to download and prepare the
    data (in `DownloadSoundata`)
    """
    task_name = self.task_config["soundata_dataset_name"]
    dataset = soundata.initialize(task_name, data_home=workdir)

    return dataset


class DownloadExtractSoundata(luigi_util.WorkTask):
    "Download and extract the Soundata dataset"

    remote = luigi.OptionalParameter()

    @property
    def stage_number(self) -> int:
        return 0

    def run(self):
        dataset = get_soundata_dataset(self.task_config, self.workdir)
        # Download and prepare the data in the task folder
        remote = self.remote or None # make sure unspecified 
        dataset.download(partial_download=remote)
        self.mark_complete()


def get_download_and_extract_tasks_soundata(
    task_config: Dict,
) -> Dict[str, luigi_util.WorkTask]:
    """Gets all the download and extract tasks for tensorflow dataset"""
    tasks = {}
    for obj in task_config["soundata_audio_remotes"]:
        outdir = obj["split"]
        tasks[outdir] = DownloadExtractSoundata(
            remote=obj["remote"],
            task_config=task_config
        )

    for obj in task_config["soundata_meta_remotes"]:
        outdir = obj["name"]
        tasks[outdir] = DownloadExtractSoundata(
            remote=obj["remote"],
            task_config=task_config
        )

    return tasks


# Define an ExtractMetadata class for each type of metadata
class ExtractSpatialEventsMetadata(ExtractMetadata):
    """
    All the splits are present in the soundata data set by default.
    If not, please override this `ExtractMetadata`, rather than using it
    as it is to extract metadata for the splits present in the data set.
    In this case, the not found splits will be automatically sampled from the
    train set in the `ExtractMetadata.split_train_test_val`.
    """

    train = luigi.TaskParameter()
    test = luigi.TaskParameter()
    valid = luigi.TaskParameter()

    def requires(self):
        # Override this depending on available splits
        return {"train": self.train, "test": self.test, "valid": self.valid}

    @staticmethod
    def skip_clip(clip, split) -> bool:
        # Override for filtering based on clip object and split
        return False

    @staticmethod
    def skip_clip_id(clip_id, split) -> bool:
        # Override for filtering based on clip id and split to avoid loading
        # the annotation
        return False

    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Start and end times are in milliseconds
        dataset = get_soundata_dataset(
            self.task_config,
            self.workdir)

        valid_azimuths = self.task_config["soundata_valid_spatial_events"]["azimuth"]
        valid_elevations = self.task_config["soundata_valid_spatial_events"]["elevation"]
        valid_distances = self.task_config["soundata_valid_spatial_events"]["distance"]

        metadatas = []
        for clip_id in dataset.clip_ids:
            # First try and filter by clip id so we don't have to load anns
            if self.skip_clip_id(clip_id, split):
                continue
            clip = dataset.clip(clip_id)
            if self.skip_clip(clip, split):
                continue

            events = clip.spatial_events
            relpath = os.path.relpath(clip.audio_path, start=self.workdir)
            
            metadata = []
            # Iterate through each label in the clip
            for label_idx, label in enumerate(events.labels):
                num_events = len(events.intervals[label_idx])
                # Iterate through each event for the label
                for ev_idx in range(num_events):
                    num_pos_steps = events.azimuths[label_idx][ev_idx].shape[0]
                    # Iterate through each positional time step in the event
                    for step_idx in range(num_pos_steps):
                        t_start = units_utils.norm_time(
                            events.intervals[label_idx][ev_idx][0] \
                                + (step_idx * events.time_step),
                            events.intervals_unit)
                        
                        if num_pos_steps > 1:
                            # If more than one step, source is moving and the
                            # annotation end time corresponds to time_step after
                            # the start time
                            t_end =  t_start + events.time_step
                        else:
                            # If there's only one step, either the source is not
                            # moving or the event only lasts for time_step
                            t_end =  events.intervals[label_idx][ev_idx][1]
                        t_end = units_utils.norm_time(
                            t_end,
                            events.intervals_unit),
                        
                        # Normalize positions if available
                        azi = ele = dist = None
                        if valid_azimuths:
                            azi = units_utils.norm_angle(
                                events.azimuths[cls_idx][ev_idx][step_idx],
                                events.azimuths_unit)
                        if valid_elevations:
                            ele = units_utils.norm_angle(
                                events.elevations[cls_idx][ev_idx][step_idx],
                                events.elevations_unit)
                        if valid_distances:
                            dist = units_utils.norm_angle(
                                events.elevations[cls_idx][ev_idx][step_idx],
                                events.elevations_unit)

                        row = (t_start, t_end, ev_idx, label) \
                            + opt_tuple(azi, valid_azimuths) \
                            + opt_tuple(ele, valid_elevations) \
                            + opt_tuple(dist, valid_distances)

                        metadata.append(row)

            metadata = pd.DataFrame(metadata,
                columns=["start", "end", "eventidx", "label"]
                        + opt_list("azimuth", valid_azimuths) \
                        + opt_list("elevation", valid_azimuths) \
                        + opt_list("distance", valid_distances) \
                        ,
            ).assign(relpath=str(relpath), split=split)
            metadatas.append(metadata)

        metadata = pd.concat(metadatas)
        return metadata
