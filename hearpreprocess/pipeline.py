"""
Generic pipelines for datasets
"""

import json
import os
import copy
import random
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union
from urllib.parse import urlparse
import warnings

import luigi
import patoolib
import numpy as np
import pandas as pd
from slugify import slugify
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import hearpreprocess.util.audio as audio_util
from hearpreprocess import __version__
from hearpreprocess.util.luigi import (
    WorkTask,
    diagnostics,
    download_file,
    new_basedir,
    safecopy,
    str2int,
)
from hearpreprocess.util.misc import first, opt_list
from hearpreprocess.util.spatial import get_spatial_columns

INCLUDE_DATESTR_IN_FINAL_PATHS = False

# Defaults for certain pipeline parameters
SPLITS = ["train", "valid", "test"]
# This percentage should not be changed as this decides
# the data in the split and hence is not a part of the data config
# We use a 80/10/10 split.
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
TRAIN_PERCENTAGE = 100 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE
TRAINVAL_PERCENTAGE = TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE


def _diagnose_split_labels(
    task_config: Dict, taskname: str, event_str: str, df: pd.DataFrame
):
    """Makes split and label diagnostics"""
    assert "split" in df.columns
    assert "label" in df.columns

    splits = df["split"].unique()
    split_file_count = df.groupby("split")["relpath"].nunique().to_dict()
    # Get fraction of rows with a particular label for each split
    split_label_frac = {
        split: split_df["label"].value_counts(normalize=True)
        for split, split_df in df.groupby("split")
    }
    # Get frequency of a particular label for each split
    split_label_freq = {
        split: list((round(split_df, 3)).to_dict().items())
        for split, split_df in split_label_frac.items()
    }
    # Get labels which are missing for a particular split
    split_label_missing = {
        split: set(df["label"].unique()) - set(labels)
        for split, labels in df.groupby("split")["label"].apply(set).to_dict().items()
    }
    for split in splits:
        diagnostics.info(
            "{} {} file count {:5s}: {}".format(
                taskname, event_str, split, split_file_count[split]
            )
        )
    for split in splits:
        diagnostics.info(
            "{} {} label freq (descending) {:5s}: {}".format(
                taskname, event_str, split, split_label_freq[split]
            )
        )
    for split in splits:
        diagnostics.info(
            "{} {} label freq (alphabetical) {:5s}: {}".format(
                taskname, event_str, split, sorted(split_label_freq[split])
            )
        )

    for split in splits:
        if split_label_missing[split]:
            diagnostics.info(
                "{} {} MISSING LABELS {:5s}: {}".format(
                    taskname, event_str, split, split_label_missing[split]
                )
            )

    # TODO: Diagnose spatial distribution

    # Confirm that there are examples for all class labels
    # This is a requirement for many metrics
    if task_config["mode"] != "small" and any(
        split_label_missing[split] for split in splits
    ):
        # Event multilabel and SELD tasks are an exception (e.g. maestro)
        if (
            task_config["embedding_type"] == "event"
            and task_config["prediction_type"] == "multilabel"
        ) or (task_config["prediction_type"] == "seld"):
            warnings.warn(
                "All labels are not present across the splits. "
                "Please check logs to see which files are missing."
            )
        else:
            raise AssertionError(
                "All labels are not present across the splits. "
                "Please check logs to see which files are missing."
            )


class DownloadCorpus(WorkTask):
    """
    Downloads from the url and saveds it in the workdir with name
    outfile
    """

    url = luigi.Parameter()
    outfile = luigi.Parameter()
    expected_md5 = luigi.Parameter()

    def run(self):
        download_file(self.url, self.workdir.joinpath(self.outfile), self.expected_md5)
        self.mark_complete()

    @property
    def stage_number(self) -> int:
        return 0


class ExtractArchive(WorkTask):
    """
    Extracts the downloaded file in the workdir(optionally in subdir inside
    workdir)

    Parameter
        infile: filename which has to be extracted from the
            download task working directory
        download(DownloadCorpus): task which downloads the corpus to be
            extracted
    Requires:
        download(DownloadCorpus): task which downloads the corpus to be
            extracted
    """

    infile = luigi.Parameter()
    download = luigi.TaskParameter(
        visibility=luigi.parameter.ParameterVisibility.PRIVATE
    )
    # outdir is the sub dir inside the workdir to extract the file.
    outdir = luigi.Parameter()

    def requires(self):
        return {"download": self.download}

    @property
    def output_path(self):
        return self.workdir.joinpath(self.outdir)

    def run(self):
        archive_path = self.requires()["download"].workdir.joinpath(self.infile)
        archive_path = archive_path.absolute()
        # https://stackoverflow.com/questions/17614467/how-can-unrar-a-file-with-python
        # Requires unrar for rar files - sudo apt install unrar
        # Also works for types supported by shutil.unpack_archive
        if self.output_path.exists():
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {self.infile} ..Please wait")
        patoolib.extract_archive(
            archive=str(archive_path), verbosity=0, outdir=str(self.output_path)
        )
        self.mark_complete()


def get_download_and_extract_tasks(task_config: Dict) -> Dict[str, WorkTask]:
    """
    Iterates over the dowload urls and builds download and extract
    tasks for them
    """

    tasks = {}
    outdirs: Set[str] = set()
    for urlobj in task_config["download_urls"]:
        split, name, url, md5 = (
            urlobj["split"],
            urlobj.get("name", None),
            urlobj["url"],
            urlobj["md5"],
        )
        filename = os.path.basename(urlparse(url).path)
        if name is not None:
            outdir = f"{split}/{name}"
        else:
            outdir = f"{split}"
        assert outdir not in outdirs, f"{outdir} in {outdirs}. If you are downloading "
        "multiple archives into one split, they should have different 'name's."
        outdirs.add(outdir)
        task = ExtractArchive(
            download=DownloadCorpus(
                url=url, outfile=filename, expected_md5=md5, task_config=task_config
            ),
            infile=filename,
            outdir=outdir,
            task_config=task_config,
        )
        tasks[slugify(outdir, separator="_")] = task

    return tasks


class ExtractMetadata(WorkTask):
    """
    This is an abstract class that extracts metadata (including labels)
    over the full dataset.
    If we detect that the original dataset doesn't have a full
    train/valid/test split, we extract 20% validation/test if it
    is missing.

    We create a metadata csv file that will be used by downstream
    luigi tasks to curate the final dataset.

    The metadata columns are:
        * relpath
            (Possible variable) location of this audio
            file relative to the Python working directory.
            WARNING: Don't use this for hashing e.g. for splitting,
            because it may vary depending upon our choice of _workdir.
            Use datapath instead.
        * datapath [DISABLED]
            Fixed unique location of this audio file
            relative to the dataset root. This can be used for hashing
            and generating splits.
        * split
            Split of this particular audio file: ['train' 'valid', 'test']
            # TODO: Verify this
        * label
            Label for the scene or event. For multilabel, if
            there are multiple labels, they will be on different rows
            of the df.
        * start, end
            Start time in milliseconds for the event with
            this label. Event prediction tasks only, i.e. timestamp
            embeddings.
        * unique_filestem
            These filenames are used for final HEAR audio files,
            WITHOUT extension. This is because the audio might be
            in different formats, but ultimately we will convert
            it to wav.
            They must be unique across all relpaths, over the
            *entire* corpus. (Thus they imply a particular split.)
            They should be fixed across each run of this
            preprocessing pipeline.
        * split_key - See get_split_key [TODO: Move here]
    """

    outfile = luigi.Parameter()

    """
    You should define one for every (split, name) task.
    `ExtractArchive` is usually enough.

    However, custom downstream processing may be required. For
    example, `speech_commands.GenerateTrainDataset` adds silence
    and background noise instances to the train split.  Custom
    downstream tasks beyond ExtractArchive should have `output_path`
    property, like `self.ExtractArchive` or
    `speech_commands.GenerateTrainDataset`

    e.g.
    """
    # train = luigi.TaskParameter()
    # test = luigi.TaskParameter()

    def requires(self):
        # You should have one for each TaskParameter above. e.g.
        # return { "train": self.train, "test": self.test }
        ...

    @staticmethod
    def relpath_to_unique_filestem(relpath: str) -> str:
        """
        Convert a relpath to a unique filestem.
        Default: The relpath's filestem.
        Override: e.g. for speech commands, we include the command
        (parent directory) name so as not to clobber filestems.
        """
        return Path(relpath).stem

    @staticmethod
    def get_split_key(df: pd.DataFrame) -> pd.Series:
        """
        Gets the split key for each audio file.

        A file should only be in one split, i.e. we shouldn't spread
        file events across splits. This is the default behavior, and
        the split key is the filename itself.
        We use unique_filestem because it is fixed for a particular
        archive.
        (We could also use datapath.)

        Override: For some corpora:new_split_kfold
        * An instrument cannot be split (nsynth)
        * A speaker cannot be split (speech_commands)
        """
        return df["unique_filestem"]

    def get_stratify_keys(self, df: pd.DataFrame, split_keys: Sequence[str]) -> pd.Series:
        """
        Gets the stratify key for each audio file.

        A file should only be in one split, i.e. we shouldn't spread
        file events across splits. This is the default behavior, and
        the split key is the filename itself.
        We use unique_filestem because it is fixed for a particular
        archive.
        (We could also use datapath.)

        Override: For some corpora:new_split_kfold
        * An instrument cannot be split (nsynth)
        * A speaker cannot be split (speech_commands)
        """

        # Only use relevant fields, drop duplicates, and set the index to the 
        # split key for getting the stratify keys in the correct order later.
        # Note that we require that each split key have a unique set of values
        # for the stratify keys
        df = (
            df[["split_key"] + list(self.task_config["stratify_fields"])]
                .drop_duplicates()
                .set_index("split_key", verify_integrity=True)
        )
        # Build the stratify key from the fields
        df["stratify_key"] = ""
        for stratify_field in self.task_config["stratify_fields"]:
            df["stratify_key"] += "-" + df[stratify_field]
        # Return the stratify keys ordered by the provided split keys
        return df.loc[split_keys]["stratify_key"]

    def get_requires_metadata(self, requires_key: str) -> pd.DataFrame:
        """
        For a particular key in the task requires (e.g. "train", or "train_eval"),
        return a metadata dataframe with the following columns (see above):
            * relpath
            * split
            * label
            * start, end: Optional
            * eventidx: Optional
            * trackidx: Optional
            * azimuth, elevation, distance: Optional
            * azimuthleft, azimuthright: Optional
            * city, location_id: Optional
        """
        raise NotImplementedError("Deriving classes need to implement this")

    def get_all_metadata(self) -> pd.DataFrame:
        """
        Combine all metadata for this task. Should have the same
        columns described in `self.get_requires_metadata`.

        By default, we do one required task at a time and then
        concat them.

        Override: When a split cannot be computed
        using just one dataset path, and multiple datasets
        must be combined (see speech_commands).
        If you override this, make sure to `.reset_index(drop=True)`
        on the final df. You won't need to override
        `get_requires_metadata`.
        """
        metadata = pd.concat(
            [
                self.get_requires_metadata_check(requires_key)
                for requires_key in list(self.requires().keys())
            ]
        ).reset_index(drop=True)
        return metadata

    # ################  You don't need to override anything else

    def postprocess_all_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        * Assign columns unique_filestem and split_key
        * Check uniqueness of unique_filestem
        * Deterministically shuffle the metadata rows
        * If --small, keep only metadata belonging to audio
        files in the small corpus.
        """

        # tqdm.pandas()
        metadata = metadata.assign(
            # This one apply is slow for massive datasets like nsynth
            # So we disable it because datapath isn't currently used.
            # datapath=lambda df: df.relpath.progress_apply(self.relpath_to_datapath),
            unique_filestem=lambda df: df.relpath.apply(
                self.relpath_to_unique_filestem
            ),
            split_key=self.get_split_key,
        )

        # No slashes can be present in the filestems. They are files, not dirs.
        assert not metadata["unique_filestem"].str.contains("/", regex=False).any()

        # Check if one unique_filestem is associated with only one relpath.
        assert metadata["relpath"].nunique() == metadata["unique_filestem"].nunique(), (
            f'{metadata["relpath"].nunique()} != '
            + f'{metadata["unique_filestem"].nunique()}'
        )
        # Also implies there is a one to one correspondence between relpath
        # and unique_filestem.
        #  1. One unique_filestem to one relpath -- the bug which
        #    we were having is one unique_filestem for two relpath(relpath
        #    with -6 as well as +6 having the same unique_filestem),
        #    groupby by unique_filestem and see if one relpath is
        #    associated with one unique_filestem - this is done in the
        #    assert statement.
        #  2. One relpath to one unique_filestem -- always the case
        #  3. relpath.nunique() == unique_filestem.nunique(), automatically
        # holds if the above two holds.
        assert (
            metadata.groupby("unique_filestem")["relpath"].nunique() == 1
        ).all(), "One unique_filestem is associated with more than one relpath "
        "Please make sure unique_filestems are unique"

        if "datapath" in metadata.columns:
            # If you use datapath, previous assertions check its
            # uniqueness wrt relpaths.
            assert metadata["relpath"].nunique() == metadata["datapath"].nunique()
            assert (
                metadata.groupby("datapath")["relpath"].nunique() == 1
            ).all(), "One datapath is associated with more than one relpath "
            "Please make sure datapaths are unique"

        # First, put the metadata into a deterministic order.
        if "start" in metadata.columns:
            columns = ["unique_filestem", "start", "end", "label"]

            if self.task_config.get("multitrack", False):
                columns += ["trackidx"]
            if self.task_config["prediction_type"] == "seld":
                # add spatial columns (which should only be for event type)
                columns += ["eventidx"] + get_spatial_columns(self.task_config)
            elif self.task_config["prediction_type"] == "avoseld_multiregion":
                pointwise = self.task_config["spatial_projection"] == "video_azimuth_region_pointwise"
                boxwise = self.task_config["spatial_projection"] == "video_azimuth_region_boxwise"
                columns += (
                    ["trackidx"]
                    + opt_list("azimuth", pointwise)
                    + opt_list("azimuthleft", boxwise)
                    + opt_list("azimuthright", boxwise)
                )

            metadata.sort_values(
                columns, inplace=True, kind="stable",
            )
        else:
            metadata.sort_values(
                ["unique_filestem", "label"], inplace=True, kind="stable"
            )

        # Now, deterministically shuffle the metadata
        # If we are going to drop things or subselect, we don't
        # want to do it according to alphabetical or filesystem order.
        metadata = metadata.sample(
            frac=1, random_state=str2int("postprocess_all_metadata")
        ).reset_index(drop=True)

        # Filter the files which actually exist in the data
        exists = metadata["relpath"].apply(lambda relpath: Path(relpath).exists())

        # If any of the audio files in the metadata is missing, raise an error for the
        # regular dataset. However, in case of small dataset, this is expected and we
        # need to remove those entries from the metadata
        if sum(exists) < len(metadata):
            if self.task_config["mode"] == "small":
                print(
                    "All files in metadata do not exist in the dataset. This is "
                    "expected behavior when small task is running.\n"
                    f"Removing {len(metadata) - sum(exists)} entries in the "
                    "metadata"
                )
                metadata = metadata.loc[exists]
                metadata.reset_index(drop=True, inplace=True)
                assert len(metadata) == sum(exists)
            else:
                from hearpreprocess.util.misc import first
                raise FileNotFoundError(
                    "Files in the metadata are missing in the directory"
                )
        return metadata
    
    def split_train_test_val(self, metadata: pd.DataFrame, stratified: bool = False):
        """
        This functions splits the metadata into test, train and valid from train
        split if any of test or valid split is not found. We split
        based upon the split_key (see above).

        If there is any data specific split, that will already be
        done in get_all_metadata. This function is for automatic
        splitting if the splits are not found.

        Note that all files are shuffled and we pick exactly as
        many as we want for each split. Unlike using modulus of the
        hash of the split key (Google `which_set` method), the
        filename does not uniquely determine the split, but the
        entire set of audio data determines the split.
        * The downside is that if a later version of the
        dataset is released with more files, this method will not
        preserve the split across dataset versions.
        * The benefit is that, for small datasets, it correctly
        stratifies the data according to the desired percentages.
        For small datasets, unless the splits are predetermined
        (e.g. in a key file), using the size of the data set to
        stratify is unavoidable. If we do want to preserve splits
        across versions, we can create key files for audio files
        that were in previous versions.

        Three cases might arise -
        1. Validation split not found - Train will be split into valid and train
        2. Test split not found - Train will be split into test and train
        3. Validation and Test split not found - Train will be split into test, train
            and valid
        """

        splits_present = metadata["split"].unique()

        # The metadata should at least have the train split
        # test and valid if not found in the metadata can be sampled
        # from the train
        assert "train" in splits_present, "Train split not found in metadata"
        splits_to_sample = set(SPLITS).difference(splits_present)
        diagnostics.info(
            f"{self.longname} Splits not already present in the dataset, "
            + f"now sampled with split key are: {splits_to_sample}"
        )

        if "split_percentage" in self.task_config:
            orig_train_percentage = self.task_config["split_percentage"]["train"]
            orig_valid_percentage = self.task_config["split_percentage"]["valid"]
            orig_test_percentage = self.task_config["split_percentage"]["test"]
        else:
            orig_train_percentage = TRAIN_PERCENTAGE
            orig_valid_percentage = VALIDATION_PERCENTAGE
            orig_test_percentage = TEST_PERCENTAGE

        train_percentage: float
        valid_percentage: float
        test_percentage: float

        # If we want a 60/20/20 split, but we already have test and don't
        # to partition one, we want to do a 75/25/0 split. i.e. we
        # keep everything summing to one and the proportions the same.
        if splits_to_sample == set():
            return metadata
        if splits_to_sample == set(["valid"]):
            tot = (orig_train_percentage + orig_valid_percentage) / 100
            train_percentage = orig_train_percentage / tot
            valid_percentage = orig_valid_percentage / tot
            test_percentage = 0
        elif splits_to_sample == set(["test"]):
            tot = (orig_train_percentage + orig_test_percentage) / 100
            train_percentage = orig_train_percentage / tot
            valid_percentage = 0
            test_percentage = orig_test_percentage / tot
        else:
            assert splits_to_sample == set(["valid", "test"])
            train_percentage = orig_train_percentage
            valid_percentage = orig_valid_percentage
            test_percentage = orig_test_percentage
        assert (
            train_percentage + valid_percentage + test_percentage == 100
        ), f"{train_percentage + valid_percentage + test_percentage} != 100"

        diagnostics.info(
            f"{self.longname} Split percentage for splitting the train set to "
            "generate other sets: "
            "{}".format(
                {
                    "test": test_percentage,
                    "train": train_percentage,
                    "valid": valid_percentage,
                }
            )
        )

        # Deterministically sort all unique split_keys.
        split_keys = sorted(metadata[metadata.split == "train"]["split_key"].unique())
        # Deterministically shuffle all unique split_keys.
        rng_seed = "split_train_test_val"
        rng = random.Random(rng_seed)
        rng.shuffle(split_keys)
        n = len(split_keys)

        n_valid = int(round(n * valid_percentage / 100))
        n_test = int(round(n * test_percentage / 100))
        assert n_valid > 0 or valid_percentage == 0
        assert n_test > 0 or test_percentage == 0
        if not stratified:
            valid_split_keys = set(split_keys[:n_valid])
            test_split_keys = set(split_keys[n_valid : n_valid + n_test])
        else:
            stratify_keys = self.get_stratify_keys(metadata, split_keys=split_keys)
            n_eval = n_valid + n_test
            
            # Get stratified train/eval split
            sss_train_eval = StratifiedShuffleSplit(n_splits=1, test_size=n_eval, random_state=str2int(rng_seed))
            _, eval_idxs = first(sss_train_eval.split(split_keys, stratify_keys))
            eval_split_keys = split_keys[eval_idxs]
            eval_stratify_keys = stratify_keys[eval_idxs]

            # From eval split, get stratified valid/test split
            sss_valid_test = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=str2int(rng_seed))
            valid_idxs, test_idxs = first(sss_valid_test.split(eval_split_keys, eval_stratify_keys))

            valid_split_keys = eval_split_keys[valid_idxs]
            test_split_keys = eval_split_keys[test_idxs]

        metadata.loc[metadata["split_key"].isin(valid_split_keys), "split"] = "valid"
        metadata.loc[metadata["split_key"].isin(test_split_keys), "split"] = "test"
        return metadata

    def assert_correct_kfolds(self, metadata: pd.DataFrame):
        """
        Raises an AssertionError if the set of fold names in task_config doesn't
        match the set of fold names used for splits in the metadata
        """
        if set(metadata["split"].unique()) != set(self.task_config["splits"]):
            raise AssertionError(
                "Names of splits in metadata don't match the required names for a "
                f"kfold dataset. Expected: {self.task_config['splits']}. "
                f"Received: {metadata['split'].unique()}."
            )

    def split_k_folds(self, metadata: pd.DataFrame, stratified: bool = False):
        """
        Deterministically split dataset into k-folds
        """
        SEED_RETRIES = 1000
        # Deterministically sort all unique split_keys.
        split_keys = sorted(metadata["split_key"].unique())

        # A set of seeds will be tried to ensure each fold has at least
        # one example for each label
        # Since seeds are tried sequentially, this is still deterministic
        # The first seed remains "split_k_folds" to maintain consistency with
        # previous versions
        seeds = ["split_k_folds"] + [
            f"split_k_folds_retry_{i}" for i in range(SEED_RETRIES - 1)
        ]

        for i, seed in enumerate(seeds):
            # Deterministically shuffle all unique split_keys.
            rng = random.Random(seed)
            shuffled_split_keys = copy.deepcopy(split_keys)
            rng.shuffle(shuffled_split_keys)

            # Equally split the split_keys into k folds and label accordingly
            k_folds = self.task_config["nfolds"]
            if not stratified:
                folds_keys = np.array_split(shuffled_split_keys, k_folds)
            else:
                stratify_keys = self.get_stratify_keys(metadata, split_keys=split_keys)
                shuffled_split_keys = np.array(shuffled_split_keys)
                kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=str2int(seed))
                folds_keys = [
                    shuffled_split_keys[np.array(fold_idxs, dtype=int)]
                    for _, fold_idxs in kf.split(shuffled_split_keys, stratify_keys)
                ]

            for j, fold in enumerate(folds_keys):
                metadata.loc[metadata["split_key"].isin(fold), "split"] = f"fold{j:02d}"

            if self.task_config["mode"] == "small" or (
                self.task_config["embedding_type"] == "event"
                and self.task_config["prediction_type"] == "multilabel"
            ) or (
                self.task_config["prediction_type"] == "seld" # TODO: could we add PSLA/DPP sampling?
            ):
                # All labels are not required across all folds for either small
                # or an event and multilable type task
                break

            # TODO: balance for spatial
            if all(
                metadata.groupby("split")["label"].nunique()
                == metadata["label"].nunique()
            ):
                diagnostics.info(
                    f"{self.longname} - Kfold was successful with seed: {seed}"
                )
                break

        if i == len(seeds) - 1:
            raise AssertionError(
                "All the seeds were tried, but the data couldnot be split into "
                "folds with at least one example for each label"
            )
        return metadata

    def create_splits(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the dataset based on the split mode in task config.
        Either train/validation/test split or a k-fold split
        """
        if self.task_config["split_mode"] == "trainvaltest":
            # Split the metadata to create valid and test set from train if
            # they are not created explicitly in get_all_metadata
            if set(self.task_config["splits"]) != set(SPLITS):
                raise AssertionError(f"Splits for trainvaltest mode must be {SPLITS}")
            metadata = self.split_train_test_val(metadata)
        elif self.task_config["split_mode"] == "stratified_trainvaltest":
            # Split the metadata to create valid and test set from train if
            # they are not created explicitly in get_all_metadata
            if set(self.task_config["splits"]) != set(SPLITS):
                raise AssertionError(f"Splits for trainvaltest mode must be {SPLITS}")
            for k in self.task_config["stratify_fields"]:
                if not k in metadata.keys():
                    raise AssertionError(
                        f"Stratify field {k} not found in metadata columns"
                    )
            metadata = self.split_train_test_val(metadata, stratified=True)
        elif self.task_config["split_mode"] == "presplit_kfold":
            # Splits are the predefined folds in the dataset - just make sure the
            # names are correct
            self.assert_correct_kfolds(metadata)
        elif self.task_config["split_mode"] == "new_split_kfold":
            # Split the dataset into k-folds
            metadata = self.split_k_folds(metadata)
            self.assert_correct_kfolds(metadata)
        elif self.task_config["split_mode"] == "new_split_stratified_kfold":
            for k in self.task_config["stratify_fields"]:
                if not k in metadata.keys():
                    raise AssertionError(
                        f"Stratify field {k} not found in metadata columns"
                    )
            # Split the dataset into k-folds
            metadata = self.split_k_folds(metadata, stratified=True)
            self.assert_correct_kfolds(metadata)
        else:
            raise ValueError("Unknown split_mode received in task_config")

        return metadata

    def trim_event_metadata(self, metadata: pd.DataFrame, duration: float):
        """
        This modifies the event metadata to
        retain only events which start before the sample duration
        and trim them if they extend beyond the sample duration
        """
        # Since the duration in the task config is in seconds convert to milliseconds
        duration_ms = duration * 1000.0
        assert "start" in metadata.columns
        assert "end" in metadata.columns

        # Drop the events starting at or after the sample duration
        trimmed_metadata = metadata.loc[metadata["start"] < duration_ms]
        events_dropped = len(metadata) - len(trimmed_metadata)

        # Trim the events starting before but extending beyond the sample duration
        events_trimmed = len(trimmed_metadata.loc[metadata["end"] > duration_ms])
        trimmed_metadata.loc[metadata["end"] > duration_ms, "end"] = duration_ms

        assert (trimmed_metadata["start"] < duration_ms).all()
        assert (trimmed_metadata["end"] <= duration_ms).all()
        assert len(trimmed_metadata) <= len(metadata)
        assert (
            metadata["relpath"].nunique() == trimmed_metadata["relpath"].nunique()
        ), "File are getting removed while trimming. This is "
        "unexpected and only events from the end of the files should be removed"

        diagnostics.info(
            f"{self.longname} - Events dropped count {events_dropped} "
            "percentage {}%".format(round(events_dropped / len(metadata) * 100.0, 2))
        )
        diagnostics.info(
            f"{self.longname} - Events trimmed count {events_trimmed} "
            "percentage {}%".format(round(events_dropped / len(metadata) * 100.0, 2))
        )
        return trimmed_metadata

    def get_requires_metadata_check(self, requires_key: str) -> pd.DataFrame:
        df = self.get_requires_metadata(requires_key)
        assert "relpath" in df.columns
        assert "split" in df.columns
        assert "label" in df.columns
        if self.task_config["embedding_type"] == "event":
            assert "start" in df.columns
            assert "end" in df.columns
        if self.task_config["prediction_type"] == "seld":
            assert "eventidx" in df.columns
            assert (
                not self.task_config.get("multitrack")
                or "trackidx" in df.columns
            )
            for col in get_spatial_columns(self.task_config):
                assert col in df.columns
        if self.task_config["prediction_type"] == "avoseld_multiregion":
            assert "trackidx" in df.columns
            if self.task_config["spatial_projection"] == "video_azimuth_region_pointwise":
                assert "azimuth" in df.columns
            if self.task_config["spatial_projection"] == "video_azimuth_region_boxwise":
                assert "azimuthleft" in df.columns
                assert "azimuthright" in df.columns
        return df

    def run(self):
        # Output stats for every input directory
        for key, requires in self.requires().items():
            stats = audio_util.get_audio_dir_stats(
                in_dir=requires.output_path,
                out_file=self.workdir.joinpath(f"{key}_stats.json"),
            )
            diagnostics.info(f"{self.longname} extractdir {key} stats {stats}")

        # Get all metadata to be used for the task
        metadata = self.get_all_metadata()
        print(f"metadata length = {len(metadata)}")

        _diagnose_split_labels(self.task_config, self.longname, "original", metadata)
        metadata = self.postprocess_all_metadata(metadata)
        _diagnose_split_labels(
            self.task_config, self.longname, "postprocessed", metadata
        )

        # Creates splits based on the split_mode for this dataset
        # Either train/val/test split or a k-fold split strategy
        metadata = self.create_splits(metadata)

        # Each split should have unique files and no file should be across splits
        assert (
            metadata.groupby("unique_filestem")["split"].nunique() == 1
        ).all(), "One unique_filestem is associated with more than one split"

        _diagnose_split_labels(self.task_config, self.longname, "split", metadata)

        if self.task_config["embedding_type"] == "scene":
            # Multiclass predictions should only have a single label per file
            if self.task_config["prediction_type"] == "multiclass":
                label_count = metadata.groupby("unique_filestem")["label"].aggregate(
                    len
                )
                assert (label_count == 1).all()
        elif self.task_config["embedding_type"] == "event":
            # Remove the events starting after the sample duration, and trim
            # the events starting before but extending beyond the sample
            # duration
            # sample duration is specified in the task config.
            # The specified sample duration is in seconds

            # If the sample duration is set to None, no trimming of events will
            # be done and the full audio file will be selected. This mode is
            # only for special tasks and should not be generally used.
            # Having all the audio files of the same length is more
            # efficient for downstream pipelines

            if self.task_config["sample_duration"] is not None:
                metadata = self.trim_event_metadata(
                    metadata, duration=self.task_config["sample_duration"]
                )
        else:
            raise ValueError(
                "%s embedding_type unknown" % self.task_config["embedding_type"]
            )

        metadata.to_csv(
            self.workdir.joinpath(self.outfile),
            index=False,
        )

        # Save the label count for each split
        for split, split_df in metadata.groupby("split"):
            json.dump(
                split_df["label"].value_counts(normalize=True).to_dict(),
                self.workdir.joinpath(f"labelcount_{split}.json").open("w"),
                indent=True,
            )

        self.mark_complete()

    # UNUSED
    def relpath_to_datapath(self, relpath: Path) -> Path:
        """
        Given the path to this audio file from the Python working
        directory, strip all output_path from each required task.

        This filename directory is a little fiddly and gnarly.
        """
        # Find all possible base paths into which audio was extracted
        base_paths = [t.output_path for t in self.requires().values()]
        assert len(base_paths) == len(set(base_paths)), (
            "You seem to have duplicate (split, name) in your task "
            + f"config. {len(base_paths)} != {len(set(base_paths))}"
        )
        datapath = Path(relpath)
        relatives = 0
        for base_path in base_paths:
            if datapath.is_relative_to(base_path):  # type: ignore
                datapath = datapath.relative_to(base_path)
                relatives += 1
        assert relatives == 1, f"relatives {relatives}. " + f"base_paths = {base_paths}"
        assert datapath != relpath, (
            f"datapath in {relpath} not found. " + f"base_paths = {base_paths}"
        )
        return datapath


class MetadataTask(WorkTask):
    """
    Abstract WorkTask that wants to have access to the metadata
    from the entire dataset.
    """

    metadata_task: ExtractMetadata = luigi.TaskParameter()
    _metadata: Optional[pd.DataFrame] = None

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = pd.read_csv(
                self.metadata_task.workdir.joinpath(self.metadata_task.outfile),
                dtype={"relpath": str, "unique_filestem": str, "split": str},
            )
        return self._metadata


class SplitTask(MetadataTask):
    """
    Abstract MetadataTask that has a split directory of workdir, and
    split parameter.
    """

    split = luigi.Parameter()

    @property
    def splitdir(self) -> Path:
        return self.workdir.joinpath(self.split)

    def createsplit(self):
        # Would be nice to have this happen automatically
        if self.splitdir.exists():
            assert self.splitdir.is_dir()
            shutil.rmtree(self.splitdir)
        self.splitdir.mkdir()


class SubsampleSplit(SplitTask):
    """
    For large datasets, we may want to restrict each split to a
    certain number of minutes.
    This subsampler acts on a specific split, and ensures we are under
    our desired audio length threshold for this split.

    Parameters:
        split: name of the split for which subsampling has to be done
    """

    def requires(self):
        return {
            "metadata": self.metadata_task,
        }

    def get_max_split_duration(self) -> Union[float, int, None]:
        """
        Returns the max duration for the current split from the task_config
        """
        # Key to use to get the max task duration depending on the split mode
        if self.task_config["split_mode"] == "trainvaltest":
            assert "max_task_duration_by_split" in self.task_config
            max_durations = self.task_config["max_task_duration_by_split"]
            if set(max_durations.keys()) != set(self.task_config["splits"]):
                raise AssertionError(
                    "Max duration must be specified for all splits/folds in "
                    "task_config, or set to None to use the full length."
                    f"Expected: {set(self.task_config['splits'])}, received:"
                    f"{set(max_durations.keys())}."
                )
            max_split_duration = max_durations[self.split]
        else:
            assert "max_task_duration_by_fold" in self.task_config
            max_split_duration = self.task_config["max_task_duration_by_fold"]

        return max_split_duration

    def run(self):
        self.createsplit()

        split_metadata = self.metadata[
            self.metadata["split"] == self.split
        ].reset_index(drop=True)
        # Get all unique_filestem in this split, deterministically sorted
        split_filestem_relpaths = (
            split_metadata[["unique_filestem", "relpath"]]
            .drop_duplicates()
            .sort_values("unique_filestem")
        )

        assert split_filestem_relpaths["relpath"].nunique() == len(
            split_filestem_relpaths
        )
        # Deterministically shuffle the filestems
        split_filestem_relpaths = split_filestem_relpaths.sample(
            frac=1, random_state=str2int(f"SubsampleSplit({self.split})")
        ).values

        # This might round badly for small corpora with long audio :\
        # But we aren't going to use audio that is more than a couple
        # minutes or the timestamp embeddings will explode
        sample_duration = self.task_config["sample_duration"]
        max_split_duration = self.get_max_split_duration()
        if sample_duration is None:
            assert max_split_duration is None, (
                "If the sample duration is set to None i.e. orignal audio files "
                "are being used without any trimming or padding, then the "
                "max_split_duration should also be None, so that no "
                "subsampling is done as the audio file length is not "
                "consistent."
            )

        # If max_split_duration is not None set the max_files so that
        # the total duration of all the audio files after subsampling
        # comes around (less than) max_split_duration
        if max_split_duration is not None:
            max_files = int(max_split_duration / sample_duration)
        else:
            # Otherwise set max_files to select all the files in the split
            max_files = len(split_filestem_relpaths)

        diagnostics.info(
            f"{self.longname} "
            f"Max number of files to sample in split {self.split}: "
            f"{max_files}"
        )

        diagnostics.info(
            f"{self.longname} "
            f"Files in split {self.split} before resampling: "
            f"{len(split_filestem_relpaths)}"
        )
        split_filestem_relpaths = split_filestem_relpaths[:max_files]
        diagnostics.info(
            f"{self.longname} "
            f"Files in split {self.split} after resampling: "
            f"{len(split_filestem_relpaths)}"
        )

        for unique_filestem, relpath in split_filestem_relpaths:
            audiopath = Path(relpath)
            newaudiofile = Path(
                # Add the current filetype suffix (mp3, webm, etc)
                # to the unique filestem.
                self.splitdir.joinpath(unique_filestem + audiopath.suffix)
            )
            assert not newaudiofile.exists(), f"{newaudiofile} already exists! "
            "We shouldn't have two files with the same name. If this is happening "
            "because luigi is overwriting an incomplete output directory "
            "we should write code to delete the output directory "
            "before this tasks begins."
            "If this is happening because different data dirs have the same "
            "audio file name, we should include the data dir in the symlinked "
            "filename."
            newaudiofile.symlink_to(audiopath.resolve())

        self.mark_complete()


class WavSplit(SplitTask):
    """
    Converts the files to wav. 

    This task ensures that the audio is converted to an uncompressed
    format so that any downstream operation on the audio results
    in precise outputs (like trimming and padding)
    https://stackoverflow.com/questions/54153364/ffmpeg-being-inprecise-when-trimming-mp3-files # noqa: E501

    Requires:
        corpus (SubsampleSplit): Split that was subsampled.
    """

    def requires(self):
        return {
            "corpus": SubsampleSplit(
                split=self.split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
        }

    def run(self):
        self.createsplit()

        for audiofile in tqdm(list(self.requires()["corpus"].splitdir.iterdir())):
            if audiofile.suffix == ".json":
                continue
            newaudiofile = self.splitdir.joinpath(f"{audiofile.stem}.wav")
            audio_util.to_wav(str(audiofile), str(newaudiofile))

        self.mark_complete()


class TrimPadSplit(SplitTask):
    """
    Trims and pads the wav audio files

    Requires:
        corpus (WavSplit): task which converts the audio to wav file
    """

    def requires(self):
        return {
            "corpus": WavSplit(
                split=self.split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
        }

    def run(self):
        self.createsplit()
        for audiofile in tqdm(list(self.requires()["corpus"].splitdir.iterdir())):
            newaudiofile = self.splitdir.joinpath(f"{audiofile.stem}.wav")
            if self.task_config["sample_duration"] is not None:
                audio_util.trim_pad_wav(
                    str(audiofile),
                    str(newaudiofile),
                    duration=self.task_config["sample_duration"],
                )
            else:
                # If the sample_duration is None, the file will be copied
                # without any trimming or padding
                safecopy(src=audiofile, dst=newaudiofile)
        self.mark_complete()


class SubcorpusData(MetadataTask):
    """
    Aggregates subsampling of all the splits into a single task as dependencies.

    Requires:
        splits (list(SplitTask)): final task over each split.
            This is a TrimPadSplit.
    """

    def requires(self):
        # Perform subsampling on each split independently
        splits = {
            split: TrimPadSplit(
                split=split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
            for split in self.task_config["splits"]
        }
        return splits

    def run(self):
        workdir = Path(self.workdir)
        if workdir.exists():
            if workdir.is_dir():
                workdir.rmdir()
            else:
                workdir.unlink()

        # We need to link the workdir of the requires, they will all be the same
        # for all the requires so just grab the first one.
        key = list(self.requires().keys())[0]
        workdir.symlink_to(Path(self.requires()[key].workdir).absolute())
        for key2 in self.requires().keys():
            assert self.requires()[key].workdir == self.requires()[key2].workdir

        # Output stats for every input directory
        for split in self.task_config["splits"]:
            stats = audio_util.get_audio_dir_stats(
                in_dir=self.workdir.joinpath(split),
                out_file=self.workdir.joinpath(f"{split}_stats.json"),
            )
            diagnostics.info(f"{self.longname} {split} stats {stats}")

        self.mark_complete()


class SubcorpusMetadata(MetadataTask):
    """
    Find the metadata for the subcorpus, based upon which audio
    files are in each subcorpus split.

    Requires
        data (SubcorpusData): which produces the subcorpus data.
    """

    def requires(self):
        return {
            "data": SubcorpusData(
                metadata_task=self.metadata_task, task_config=self.task_config
            ),
        }

    def run(self):
        split_label_dfs = []
        for split in self.task_config["splits"]:
            split_path = self.requires()["data"].workdir.joinpath(split)
            audiodf = pd.DataFrame(
                [(a.stem, a.suffix) for a in list(split_path.glob("*.wav"))],
                columns=["unique_filestem", "ext"],
            )
            assert len(audiodf) != 0, f"No audio files found in: {split_path}"
            assert (
                not audiodf["unique_filestem"].duplicated().any()
            ), "Duplicate files in: {split_path}"
            assert len(audiodf["ext"].drop_duplicates()) == 1
            assert audiodf["ext"].drop_duplicates().values[0] == ".wav"

            # Get the label from the metadata with the help
            # of the unique_filestem of the filename
            audiolabel_df = (
                self.metadata.merge(audiodf, on="unique_filestem")
                .assign(unique_filename=lambda df: df["unique_filestem"] + df["ext"])
                .drop("ext", axis=1)
            )

            if self.task_config["embedding_type"] == "scene":
                # Create a dictionary containing a list of metadata
                # keyed on the unique_filestem.
                audiolabel_json = (
                    audiolabel_df[["unique_filename", "label"]]
                    .groupby("unique_filename")["label"]
                    .apply(list)
                    .to_dict()
                )

            elif self.task_config["embedding_type"] == "event":
                # For event labeling each file will have a list of metadata
                columns = ["unique_filename", "label", "start", "end"]
                if self.task_config["prediction_type"] == "seld":
                    columns += (
                        ["eventidx"]
                        + get_spatial_columns(self.task_config)
                        + opt_list("trackidx", self.task_config.get("multitrack"))
                    )
                elif self.task_config["prediction_type"] == "avoseld_multiregion":
                    pointwise = self.task_config["spatial_projection"] == "video_azimuth_region_pointwise"
                    boxwise = self.task_config["spatial_projection"] == "video_azimuth_region_boxwise"
                    columns += (
                        ["trackidx"]
                        + opt_list("azimuth", pointwise)
                        + opt_list("azimuthleft", boxwise)
                        + opt_list("azimuthright", boxwise)
                    )
                audiolabel_json = (
                    audiolabel_df[columns]
                    .set_index("unique_filename")
                    .groupby(level=0)
                    .apply(lambda group: group.to_dict(orient="records"))
                    .to_dict()
                )
            else:
                raise ValueError("Invalid embedding_type in dataset config")

            # Save the json used for training purpose
            json.dump(
                audiolabel_json,
                self.workdir.joinpath(f"{split}.json").open("w"),
                indent=True,
            )

            # Save the slug and the label in as the split metadata
            audiolabel_df.to_csv(
                self.workdir.joinpath(f"{split}.csv"),
                index=False,
            )
            split_label_dfs.append(audiolabel_df)

        _diagnose_split_labels(
            self.task_config,
            self.longname,
            "",
            pd.concat(split_label_dfs).reset_index(drop=True),
        )
        self.mark_complete()


class MetadataVocabulary(MetadataTask):
    """
    Creates the vocabulary CSV file for a task.

    Requires
        subcorpus_metadata (SubcorpusMetadata): task which produces
            the subcorpus metadata
    """

    def requires(self):
        return {
            "subcorpus_metadata": SubcorpusMetadata(
                metadata_task=self.metadata_task, task_config=self.task_config
            )
        }

    def run(self):
        labelset = set()
        # Save statistics about each subcorpus metadata
        for split in self.task_config["splits"]:
            labeldf = pd.read_csv(
                self.requires()["subcorpus_metadata"].workdir.joinpath(f"{split}.csv")
            )
            json.dump(
                labeldf["label"].value_counts(normalize=True).to_dict(),
                self.workdir.joinpath(f"labelcount_{split}.json").open("w"),
                indent=True,
            )
            split_labelset = set(labeldf["label"].unique().tolist())
            assert len(split_labelset) != 0
            labelset = labelset | split_labelset

        # Build the label idx csv and save it
        labelcsv = pd.DataFrame(
            list(enumerate(sorted(list(labelset)))),
            columns=["idx", "label"],
        )

        labelcsv.to_csv(
            os.path.join(self.workdir, "labelvocabulary.csv"),
            columns=["idx", "label"],
            index=False,
        )

        self.mark_complete()


class ResampleSubcorpus(MetadataTask):
    """
    Resamples one split in the subsampled corpus to a particular sampling rate
    Parameters
        split (str): The split for which the resampling has to be done
        sr (int): output sampling rate
    Requires
        data (SubcorpusData): task which produces the subcorpus data
    """

    sr = luigi.IntParameter()
    split = luigi.Parameter()

    def requires(self):
        return {
            "data": SubcorpusData(
                metadata_task=self.metadata_task, task_config=self.task_config
            )
        }

    def run(self):
        original_dir = self.requires()["data"].workdir.joinpath(str(self.split))
        resample_dir = self.workdir.joinpath(str(self.sr)).joinpath(str(self.split))
        resample_dir.mkdir(parents=True, exist_ok=True)
        for audiofile in tqdm(sorted(list(original_dir.glob("*.wav")))):
            resampled_audiofile = new_basedir(audiofile, resample_dir)
            audio_util.resample_wav(audiofile, resampled_audiofile, self.sr)

        self.mark_complete()


class ChannelReformatSubcorpus(MetadataTask):
    """
    Reformats channels for audio in one split in the subsampled corpus to a
    particular channel format
    Parameters
        split (str): The split for which the resampling has to be done
        sr (int): output sampling rate
    Requires
        data (SubcorpusData): task which produces the subcorpus data
    """
    channel_format = luigi.Parameter()
    sr = luigi.IntParameter()
    split = luigi.Parameter()

    def requires(self):
        tasks = {
            "data": ResampleSubcorpus(
                sr=self.sr, split=self.split, task_config=self.task_config, metadata_task=self.metadata_task
            )
        }
        if self.task_config["in_channel_format"] == "foa" and self.channel_format == "stereo":
            tasks["vst3_task"] = audio_util.FOAToBinauralTask(
                task_config=self.task_config)

        return tasks

    def run(self):
        original_dir = self.requires()["data"].workdir.joinpath(str(self.sr)).joinpath(str(self.split))
        reformat_dir = self.workdir.joinpath(str(self.channel_format)).joinpath(str(self.sr)).joinpath(str(self.split))
        reformat_dir.mkdir(parents=True, exist_ok=True)
        for audiofile in tqdm(sorted(list(original_dir.glob("*.wav")))):
            reformated_audiofile = new_basedir(audiofile, reformat_dir)
            audio_util.channel_reformat_wav(audiofile, reformated_audiofile,
                in_chfmt=self.task_config.get("in_channel_format", "mixed_mono_stereo"),
                out_chfmt=self.channel_format,
                vst3_task=self.requires().get("vst3_task")
            )

        self.mark_complete()


class ResampleSubcorpuses(MetadataTask):
    """
    Aggregates resampling of all the splits and sampling rates
    into a single task as dependencies.

    Requires:
        ResampleSubcorpus for all split and sr
    """

    sample_rates = luigi.ListParameter()

    def requires(self):
        # Perform resampling on each split and sampling rate independently
        resample_splits = [
            ResampleSubcorpus(
                sr=sr,
                split=split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
            for sr in self.sample_rates
            for split in self.task_config["splits"]
        ]
        return resample_splits

    def run(self):
        workdir = Path(self.workdir)
        workdir.rmdir()
        # We need to link the workdir of the requires, they will all be the same
        # for all the requires so just grab the first one.
        requires_workdir = Path(self.requires()[0].workdir).absolute()
        workdir.symlink_to(requires_workdir)
        self.mark_complete()


class ChannelReformatSubcorpuses(MetadataTask):
    """
    Aggregates channel reformatting of all the splits, sampling rates, and
    formats into a single task as dependencies.

    Requires:
        ChannelReformatSubcorpus for all split and sr
    """

    sample_rates = luigi.ListParameter()
    channel_formats = luigi.ListParameter()

    def requires(self):
        # Perform resampling on each split, sampling rate, and channel
        # format independently. Dependencies set up so resampling is done
        # before channel reformatting
        resample_splits = [
            ChannelReformatSubcorpus(
                channel_format=ch_fmt,
                sr=sr,
                split=split,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
            for ch_fmt in self.channel_formats
            for sr in self.sample_rates
            for split in self.task_config["splits"]
        ]
        return resample_splits

    def run(self):
        workdir = Path(self.workdir)
        workdir.rmdir()
        # We need to link the workdir of the requires, they will all be the same
        # for all the requires so just grab the first one.
        requires_workdir = Path(self.requires()[0].workdir).absolute()
        workdir.symlink_to(requires_workdir)
        self.mark_complete()


class FinalCombine(MetadataTask):
    """
    Create a final dataset, no longer in _workdir but in directory
    tasks_dir.

    Parameters:
        sample_rates (list(int)): The list of sampling rates in
            which the corpus is required.
        tasks_dir str: Directory to put the combined dataset.
    Requires:
        resample (List(ResampleSubCorpus)): task which resamples
            the entire subcorpus
        subcorpus_metadata (SubcorpusMetadata): task with the
            subcorpus metadata
    """

    sample_rates = luigi.ListParameter()
    channel_formats = luigi.ListParameter()
    tasks_dir = luigi.Parameter()

    def requires(self):
        # Will copy the resampled subsampled data, the subsampled metadata,
        # and the metadata_vocabulary
        return {
            "resample": ResampleSubcorpuses(
                sample_rates=self.sample_rates,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            ),
            "channel_reformat": ChannelReformatSubcorpuses(
                sample_rates=self.sample_rates,
                channel_formats=self.channel_formats,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            ),
            "subcorpus_metadata": SubcorpusMetadata(
                metadata_task=self.metadata_task, task_config=self.task_config
            ),
            "metadata_vocabulary": MetadataVocabulary(
                metadata_task=self.metadata_task, task_config=self.task_config
            ),
        }

    # We overwrite workdir here, because we want the output to be
    # the finalized task directory
    @property
    def workdir(self):
        return Path(self.tasks_dir).joinpath(self.versioned_task_name)

    def run(self):
        if self.workdir.exists():
            shutil.rmtree(self.workdir)

        # Copy the resample and reformatted files
        shutil.copytree(self.requires()["channel_reformat"].workdir, self.workdir)

        # Copy labelvocabulary.csv
        shutil.copy2(
            self.requires()["metadata_vocabulary"].workdir.joinpath(
                "labelvocabulary.csv"
            ),
            self.workdir.joinpath("labelvocabulary.csv"),
        )
        # Copy the train test metadata jsons
        src = self.requires()["subcorpus_metadata"].workdir
        dst = self.workdir
        for item in sorted(os.listdir(src)):
            if item.endswith(".json"):
                # Based upon https://stackoverflow.com/a/27161799
                assert not dst.joinpath(item).exists()
                assert not src.joinpath(item).is_dir()
                shutil.copy2(src.joinpath(item), dst.joinpath(item))
        # Python >= 3.8 only
        # shutil.copytree(src, dst, dirs_exist_ok=True, \
        #        ignore=shutil.ignore_patterns("*.csv"))
        # Save the dataset config as a json file
        config_out = self.workdir.joinpath("task_metadata.json")
        with open(config_out, "w") as fp:
            json.dump(
                self.task_config, fp, indent=True, cls=luigi.parameter._DictParamEncoder
            )

        self.mark_complete()


class TarCorpus(MetadataTask):
    """
    Tar the final dataset at some sample rate.

    TODO: Secret tasks should go into another directory,
    so we don't accidentally copy them to the public bucket.

    Parameters:
        sample_rates (list(int)): The list of sampling rates in
            which the corpus is required.
        tasks_dir str: Directory to put the combined dataset.
        tar_dir str: Directory to put the tar-files.
    Requires:
        combined (FinalCombine): Final combined dataset.
    """

    sample_rate = luigi.IntParameter()
    channel_format = luigi.Parameter()
    combined_task = luigi.TaskParameter()
    tasks_dir = luigi.Parameter()
    tar_dir = luigi.Parameter()

    def requires(self):
        return {"combined": self.combined_task}

    def source_to_archive_path(
        self, source_path: Union[str, Path], datestr: str
    ) -> str:
        source_path = str(source_path)
        archive_path = source_path.replace(self.tasks_dir, "tasks").replace(
            "tasks//", "tasks/"
        )
        assert (
            self.tasks_dir in ("tasks", "tasks/") or archive_path != source_path
        ), f"{archive_path} == {source_path}"
        assert archive_path.startswith("tasks")
        archive_path = f"hear-{datestr}{__version__}/{archive_path}"
        return archive_path

    @staticmethod
    def tar_filter(tarinfo: tarfile.TarInfo, pbar: tqdm) -> Optional[tarfile.TarInfo]:
        """tarfile with progress bar"""
        pbar.update(1)
        return tarinfo

    def create_tar(self, sample_rate: int, channel_format: str):
        if INCLUDE_DATESTR_IN_FINAL_PATHS:
            datestr = datetime.today().strftime("%Y%m%d") + "-"
        else:
            datestr = ""
        tarname = (
            f"hear-{datestr}{__version__}-"
            + f"{self.versioned_task_name}-{channel_format}-{sample_rate}.tar.gz"
        )
        tarname_latest = f"hear-LATEST-{self.versioned_task_name}-{channel_format}-{sample_rate}.tar.gz"
        source_dir = str(self.requires()["combined"].workdir)

        # Compute the audio files to be tar'ed
        files = set()
        for split in self.task_config["splits"]:
            files |= set(
                json.load(open(os.path.join(source_dir, f"{split}.json"))).keys()
            )

        # tarfile is pure python and very slow
        # But it's easy to precisely control, so we use it
        tarfile_workdir = self.workdir.joinpath(tarname)
        assert not os.path.exists(tarfile_workdir)
        with tarfile.open(tarfile_workdir, "w:gz") as tar:
            # First, add all files in the task
            for source_file in Path(source_dir).glob("*"):
                if source_file.is_file():
                    tar.add(
                        source_file, self.source_to_archive_path(source_file, datestr)
                    )
            # Now add audio files for this sample rate and channel format
            channel_format_sample_rate_source = os.path.join(source_dir, channel_format, str(sample_rate))
            with tqdm(
                desc=f"tar {self.versioned_task_name} {sample_rate}", total=len(files)
            ) as pbar:
                tar.add(
                    channel_format_sample_rate_source,
                    self.source_to_archive_path(channel_format_sample_rate_source, datestr),
                    filter=lambda tarinfo: self.tar_filter(tarinfo, pbar),
                )
        shutil.copyfile(tarfile_workdir, Path(self.tar_dir).joinpath(tarname))
        shutil.copyfile(tarfile_workdir, Path(self.tar_dir).joinpath(tarname_latest))

    def run(self):
        self.create_tar(self.sample_rate, self.channel_format)
        self.mark_complete()


class FinalizeCorpus(MetadataTask):
    """
    Finalize the corpus, simply create all tar files.

    Parameters:
        sample_rates (list(int)): The list of sampling rates in
            which the corpus is required.
        tasks_dir str: Directory to put the combined dataset.
        tar_dir str: Directory to put the tar-files.
    Requires:
        combined (FinalCombine): Final combined dataset.
    """

    sample_rates = luigi.ListParameter()
    channel_formats = luigi.ListParameter()
    tasks_dir = luigi.Parameter()
    tar_dir = luigi.Parameter()

    def requires(self):
        combined_task = FinalCombine(
            sample_rates=self.sample_rates,
            channel_formats=self.channel_formats,
            tasks_dir=self.tasks_dir,
            metadata_task=self.metadata_task,
            task_config=self.task_config,
        )
        return {
            str(sr): TarCorpus(
                sample_rate=sr,
                channel_format=ch_fmt,
                combined_task=combined_task,
                tasks_dir=self.tasks_dir,
                tar_dir=self.tar_dir,
                metadata_task=self.metadata_task,
                task_config=self.task_config,
            )
            for ch_fmt in self.channel_formats
            for sr in self.sample_rates
        }

    def run(self):
        self.mark_complete()


def run(task: Union[List[luigi.Task], luigi.Task], num_workers: int):
    """
    Run a task / set of tasks

    Args:
        task: a single or list of luigi tasks
        num_workers: Number of CPU workers to use for this task
    """

    # If this is just a single task then add it to a list
    if isinstance(task, luigi.Task):
        task = [task]

    diagnostics.info("LUIGI START")
    luigi_run_result = luigi.build(
        task,
        workers=num_workers,
        local_scheduler=True,
        log_level="INFO",
        detailed_summary=True,
    )
    diagnostics.info("LUIGI END")
    assert luigi_run_result.status in [
        luigi.execution_summary.LuigiStatusCode.SUCCESS,
        luigi.execution_summary.LuigiStatusCode.SUCCESS_WITH_RETRY,
    ], f"Received luigi_run_result.status = {luigi_run_result.status}"
