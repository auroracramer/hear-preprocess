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
from typing import Dict, Optional, Type

import luigi
import pandas as pd
import soundata
import soundata.core

import hearpreprocess.pipeline as pipeline
import hearpreprocess.util.luigi as luigi_util
import hearpreprocess.util.units as units_util
from hearpreprocess.util.misc import opt_list, opt_tuple, first

logger = logging.getLogger("luigi-interface")


class DownloadExtractSoundata(luigi_util.WorkTask):
    "Download and extract the Soundata dataset"

    task_name = luigi.Parameter()
    remotes = luigi.ListParameter()
    _dataset: Optional[soundata.core.Dataset] = None

    @property
    def stage_number(self) -> int:
        return 0

    @property
    def output_path(self):
        raise NotImplementedError("Deriving classes need to implement this")

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = soundata.initialize(self.task_name, data_home=self.workdir)
        return self._dataset


    def run(self):
        # Download and prepare the data in the task folder
        remotes = list(self.remotes) if self.remotes else None
        self.dataset.download(partial_download=remotes)
        self.mark_complete()


def get_download_and_extract_tasks_soundata(
    task_config: Dict,
    download_cls: Type[DownloadExtractSoundata],
) -> Dict[str, luigi_util.WorkTask]:
    """Gets all the download and extract tasks for tensorflow dataset"""
    tasks = {}
    for obj in task_config["soundata_splits"]:
        outdir = obj["split"]
        tasks[outdir] = download_cls(
            task_name=task_config["soundata_dataset_name"],
            remotes=obj["remotes"],
            task_config=task_config
        )

    return tasks


class ExtractSoundataMetadata(pipeline.ExtractMetadata):
    """
    All the splits are present in the soundata data set by default.
    If not, please override this `ExtractMetadata`, rather than using it
    as it is to extract metadata for the splits present in the data set.
    In this case, the not found splits will be automatically sampled from the
    train set in the `ExtractMetadata.split_train_test_val`.
    """

    def requires(self):
        # Override this depending on available splits
        raise NotImplementedError("Deriving classes need to implement this")

    def _get_split_dict(self, split):
        return first(
            split_obj for split_obj in self.task_config["soundata_splits"]
            if split_obj["split"] == split
        )

    def skip_clip(self, clip, split) -> bool:
        split_dict = self._get_split_dict(split)
        res = False
        for filter_dict in split_dict["filters"]:
            if filter_dict["type"] == "clip_attr":
                attr = filter_dict["attr_name"]
                valid = False
                # Valid if one of the filters is true
                for attr_value in filter_dict["attr_value_list"]:
                    valid = valid or (getattr(clip, attr) == attr_value)
                res = not valid

        return res

    def skip_clip_id(self, clip_id, split) -> bool:
        split_dict = self._get_split_dict(split)
        res = False
        for filter_dict in split_dict["filters"]:
            if filter_dict["type"] == "clip_id_prefix":
                valid = False
                # Valid if one of the filters is true
                for prefix in filter_dict["prefix_list"]:
                    valid = valid or clip_id.startswith(prefix)
                res = not valid
        return res

    def skip_row(self, **kwargs):
        # Override to filter specific rows within a clip, specifying args
        return False

    @staticmethod
    def convert_column_to_idxs(df: pd.DataFrame, column: str) -> pd.DataFrame:
        # Ensure track idxs are ints starting from 0
        df = df.copy()
        unique_vals = sorted(df[column].unique())
        for track_idx, val in enumerate(unique_vals):
            df[df[column] == val][column] = track_idx
        df[column] = df[column].astype(int)
        return df

    def get_final_label(self, label):
        remap_dict = self.task_config.get("soundata_remap_labels", {})
        new_label = remap_dict.get(str(label))
        label = new_label or label
        return label


# Define an ExtractMetadata class for each type of metadata
class ExtractSpatialEventsMetadata(ExtractSoundataMetadata):
    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Start and end times are in milliseconds
        dataset = self.requires()[split].dataset

        valid_azimuths = self.task_config["soundata_valid_spatial_events"]["azimuth"]
        valid_elevations = self.task_config["soundata_valid_spatial_events"]["elevation"]
        valid_distances = self.task_config["soundata_valid_spatial_events"]["distance"]
        valid_track_idxs = self.task_config.get("multitrack", False)

        column_names = (
            ["start", "end", "eventidx", "label"]
            + opt_list("trackidx", valid_track_idxs)
            + opt_list("azimuth", valid_azimuths)
            + opt_list("elevation", valid_azimuths)
            + opt_list("distance", valid_distances)
        )

        metadatas = []
        for clip_id in dataset.clip_ids:
            # First try and filter by clip id so we don't have to load anns
            if self.skip_clip_id(clip_id, split):
                continue
            clip = dataset.clip(clip_id)
            if self.skip_clip(clip, split):
                continue

            events = clip.spatial_events
            relpath = clip.get_path("audio")
            num_items = len(events.labels)

            track_idx_list = getattr(
                events,
                "track_number_indices",
                ["0"] * num_items
            )
            
            metadata = []
            # Iterate through each label in the clip
            for item_idx, label in enumerate(events.labels):
                num_events = len(events.intervals[item_idx])
                track_idx = track_idx_list[item_idx]
                # Iterate through each event for the label
                for ev_idx in range(num_events):
                    num_pos_steps = events.azimuths[item_idx][ev_idx].shape[0]
                    # Iterate through each positional time step in the event
                    for step_idx in range(num_pos_steps):
                        t_start = units_util.norm_time(
                            events.intervals[item_idx][ev_idx][0] \
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
                            t_end =  events.intervals[item_idx][ev_idx][1]
                        t_end = units_util.norm_time(
                            t_end,
                            events.intervals_unit
                        )
                        
                        # Normalize positions if available
                        azi = ele = dist = None
                        if valid_azimuths:
                            if self.task_config["spatial_projection"] == "unit_yz_disc":
                                azi = 0.0
                            else:
                                azi = units_util.norm_angle(
                                    events.azimuths[item_idx][ev_idx][step_idx],
                                    events.azimuths_unit)
                        if valid_elevations:
                            if self.task_config["spatial_projection"] == "unit_xy_disc":
                                ele = 0.0
                            else:
                                ele = units_util.norm_angle(
                                    events.elevations[item_idx][ev_idx][step_idx],
                                    events.elevations_unit)
                        if valid_distances:
                            if self.task_config["spatial_projection"] in (
                                "unit_sphere", "unit_xy_disc", "unit_yz_disc"
                            ):
                                dist = 1.0
                            else:
                                dist = units_util.norm_angle(
                                    events.elevations[item_idx][ev_idx][step_idx],
                                    events.elevations_unit)

                        label = self.get_final_label(label)
                        row = (
                            (t_start, t_end, ev_idx, label)
                            + opt_tuple(track_idx, valid_track_idxs)
                            + opt_tuple(azi, valid_azimuths)
                            + opt_tuple(ele, valid_elevations)
                            + opt_tuple(dist, valid_distances)
                        )

                        if not self.skip_row(
                            t_start=t_start,
                            t_end=t_end,
                            clip_id=clip_id,
                            label=label,
                            item_idx=item_idx,
                            ev_idx=ev_idx,
                            step_idx=step_idx,
                            track_idx=track_idx,
                        ):
                            metadata.append(row)

            metadata = pd.DataFrame(metadata, columns=column_names).assign(
                relpath=str(relpath), split=split
            )
            metadatas.append(metadata)

        metadata = pd.concat(metadatas)
        if valid_track_idxs:
            metadata = self.convert_column_to_idxs(metadata, "trackidx")
        return metadata


class ExtractEventMetadata(ExtractSoundataMetadata):
    def get_requires_metadata(self, split: str) -> pd.DataFrame:
        logger.info(f"Preparing metadata for {split}")

        # Start and end times are in milliseconds
        dataset = self.requires()[split].dataset

        valid_track_idxs = self.task_config["multitrack"]

        column_names = (
            ["start", "end", "label"]
            + opt_list("trackidx", valid_track_idxs)
        )

        metadatas = []
        for clip_id in dataset.clip_ids:
            # First try and filter by clip id so we don't have to load anns
            if self.skip_clip_id(clip_id, split):
                continue
            clip = dataset.clip(clip_id)
            if self.skip_clip(clip, split):
                continue

            events = clip.events
            relpath = clip.get_path("audio")
            num_items = len(events.labels)

            track_idx_list = getattr(
                events,
                "track_number_indices",
                ["0"] * num_items
            )
            
            metadata = []
            for ev_idx, (t_start, t_end) in enumerate(events.intervals):
                t_start, t_end = (
                    units_util.norm_time(t, events.intervals_unit)
                    for t in (t_start, t_end)
                )
                label = self.get_final_label(events.labels[ev_idx])
                track_idx = track_idx_list[ev_idx]
                        
                row = (
                    (t_start, t_end, label) 
                    + opt_tuple(track_idx, valid_track_idxs)
                )

                if not self.skip_row(
                    t_start=t_start,
                    t_end=t_end,
                    clip_id=clip_id,
                    ev_idx=ev_idx,
                    track_idx=track_idx,
                    label=label,
                ):
                    metadata.append(row)

            metadata = pd.DataFrame(metadata, columns=column_names).assign(
                relpath=str(relpath), split=split
            )
            metadatas.append(metadata)

        metadata = pd.concat(metadatas)
        if valid_track_idxs:
            metadata = self.convert_column_to_idxs(metadata, "trackidx")
        return metadata
