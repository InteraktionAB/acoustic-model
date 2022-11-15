import itertools
import os
import typing

from pathlib import Path
import numpy as np

import sagemaker
import s3fs
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio

class MelAndPitchDataset(Dataset):
    """Load pitch, mel data.

    Extract pitch and mel data.

    Attributes:
        root: Data root.
        train: Flag for training.
        discrete: Unused.
        mels_dir: Path to directory where spectrograms are stored.
        units_dir: Path to directory where content units are stored.
        wavs_dir: Path to directory where wav files are stored.
        metadata: Tails.
    """
    def __init__(self, root: str, train: bool=True, discrete: bool=False) -> None:
        """Constructs MelAndPitchDataset.

        Construct the MelAndPitchDataset from root, train and bool.

        Arguments:
            root: Data root.
            train: Train/eval flag.
            discrete: Soft/discrete flag.
        
        Returns:
            Return tuple of four tensors.
        
        """
        self.root: str = root
        self.train: bool = train
        self.discrete: bool = discrete
        self.mels_dir: str = os.path.join(self.root, "mels",)
        self.units_dir: str = os.path.join(self.root, "soft",)
        self.wavs_dir: str = os.path.join(self.root, "wavs",)

        train_dev_pattern_mels: typing.Dict = {True: "train/*.npy", False: "dev/*.npy"}
        train_dev_pattern_units: typing.Dict = {True: "train/*.npy", False: "dev/*.npy"}
        train_dev_pattern_wavs: typing.Dict = {True: "train/*.wav", False: "dev/*.wav"}
        train_dev_mels: typing.Dict = {True: "train", False: "dev"}

        pattern_mels: str = train_dev_pattern_mels[self.train]
        pattern_units: str = train_dev_pattern_units[self.train]
        pattern_wavs: str = train_dev_pattern_wavs[self.train]

        files: typing.List[str] = os.listdir(os.path.join(self.mels_dir, train_dev_mels.get(self.train)))
        self.metadata: typing.List[str] = list(
            itertools.starmap(os.path.join, zip([train_dev_mels.get(self.train)] * len(files), files,)))

    def __len__(self,):
        """Get length of data.

        Get the length of the dataset.

        Args:
            None

        Returns:
            None

        """
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        mel_path = os.path.join(self.mels_dir, path)
        units_path = os.path.join(self.units_dir, path)
        wavs_path: str = os.path.join(self.wavs_dir, f"{path.split('.')[0]}.wav")

        mel = np.load(mel_path).T
        units = np.load(units_path)
        audio, sample_rate = torchaudio.load(wavs_path,)
        pitch: torch.Tensor = torchaudio.functional.detect_pitch_frequency(audio, sample_rate,)

        length = 2 * units.shape[0]

        mel = torch.from_numpy(mel[:length, :])
        mel = F.pad(mel, (0, 0, 1, 0))
        units = torch.from_numpy(units)
        if self.discrete:
            units = units.long()
        print(mel.shape, units.shape, pitch.shape)
        return mel, units, pitch

    def pad_collate(self, batch):
        mels, units, pitches = zip(*batch)

        mels, units, pitches = list(mels), list(units), list(pitches)

        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])
        pitches_lengths = torch.tensor([x.size(0) for x in pitches])

        mels = pad_sequence(mels, batch_first=True)
        units = pad_sequence(
            units, batch_first=True, padding_value=100 if self.discrete else 0
        )

        pitches = pad_sequence(pitches.T, batch_first=True,)

        return mels, mels_lengths, units, units_lengths, pitches.T, pitches_lengths


class MelDatasetOS(Dataset):

    """Fixed version for sagemaker

    This version of the dataset differs from original version by the use of
    pathlib.

    Usage:
        import acoustic

        dataset: torch.data.Dataset = MelDataset(
            root=os.environ["S3_CHANNEL_TRAINING"], train=True, discrete=False)
    """

    def __init__(self, root: str, train: bool=True, discrete=False):

        """Constructs MelDatasetOS

        Constructs the MelDatasetOS.

        Arguments:
            root: Parent directory
            train: True if training
            discrete: True if training for discete encoder

        Returns:
            None
        """

        self.root = root
        self.train = train
        self.discrete = discrete

        soft_discrete: typing.Dict = {False: "soft", True: "discrete"}
        train_dev_pattern: typing.Dict = {True: "train/*.npy", False: "dev/*.npy"}
        train_dev: typing.Dict = {True: "train", False: "dev"}

        self.mels_dir = os.path.join(root, "mels")
        self.units_dir = os.path.join(root, soft_discrete.get(self.discrete))

        pattern: str = train_dev_pattern.get(train)

        files: typing.List[str] = os.listdir(os.path.join(self.mels_dir, train_dev.get(train)))

        print(files)
        self.metadata: typing.List[str] = list(
            itertools.starmap(os.path.join, zip([train_dev.get(train)] * len(files), files)))
        print(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        mel_path = os.path.join(self.mels_dir, path)
        units_path = os.path.join(self.units_dir, path)

        mel = np.load(mel_path).T
        units = np.load(units_path)

        length = 2 * units.shape[0]

        mel = torch.from_numpy(mel[:length, :])
        mel = F.pad(mel, (0, 0, 1, 0))
        units = torch.from_numpy(units)
        if self.discrete:
            units = units.long()
        return mel, units

    def pad_collate(self, batch):
        mels, units = zip(*batch)

        mels, units = list(mels), list(units)

        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])

        mels = pad_sequence(mels, batch_first=True)
        units = pad_sequence(
            units, batch_first=True, padding_value=100 if self.discrete else 0
        )

        return mels, mels_lengths, units, units_lengths


class MelDataset(Dataset):
    def __init__(self, root: Path, train: bool = True, discrete: bool = False):
        root: Path = Path(os.environ["SM_CHANNEL_TRAINING"])
        self.discrete = discrete
        self.mels_dir = root / "mels"
        self.units_dir = root / "discrete" if discrete else root / "soft"

        pattern = "train/**/*.npy" if train else "dev/**/*.npy"
        self.metadata = [
            path.relative_to(self.mels_dir).with_suffix("")
            for path in self.mels_dir.rglob(pattern)
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        mel_path = self.mels_dir / path
        units_path = self.units_dir / path

        mel = np.load(mel_path.with_suffix(".npy")).T
        units = np.load(units_path.with_suffix(".npy"))

        length = 2 * units.shape[0]

        mel = torch.from_numpy(mel[:length, :])
        mel = F.pad(mel, (0, 0, 1, 0))
        units = torch.from_numpy(units)
        if self.discrete:
            units = units.long()
        return mel, units

    def pad_collate(self, batch):
        mels, units = zip(*batch)

        mels, units = list(mels), list(units)

        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])

        mels = pad_sequence(mels, batch_first=True)
        units = pad_sequence(
            units, batch_first=True, padding_value=100 if self.discrete else 0
        )

        return mels, mels_lengths, units, units_lengths


class MelDatasetS3(Dataset):

    """Dataset loder for AWS simple storage service

    This MelDatasetS3 is the implementation of Dataset
    for reading Mel-Spectrograms stored in an S3 bucket.

    Usage:
        import s3fs

        import acoustic

        s3_file_system: (s3fs
        .core
        .S3FileSystem) = (s3fs
        .core
        .S3FileSystem())

        mel_dataset_s3: acoustic.dataset.MelDatasetS3 = (acoustic
        .dataset
        .MelDatasetS3(
            uri="s3://sagemaker-acostic-bucket", 
            pattern="train/*.npy", 
            discrete=False,
            s3_file_system=s3_file_system))
    """

    def __init__(self, uri: str, pattern: str, s3_file_system: s3fs.core.S3FileSystem):
        
        """Constructor for MelDatasetS3

        __init__ constructs the objects with uri of the bucket, pattern of the filename, S3FileSystem object.
        """

        self.metadata_mels: typing.List[str] = (s3_file_system
        .glob(sagemaker.s3.s3_path_join(
            uri,
            "mels", 
            pattern)))

        self.metadata_units: typing.List[str] = (s3_file_system
        .glob(sagemaker.s3.s3_path_join(
            uri,
            "soft",
            pattern
        )))

    def __len__(self) -> int:

        """Length of dataset
        
        Returns the total length of the data.

        Args:
            uri (str): URI of the bucket.
            pattern (str): Pattern of the filepath in the bucket.
            s3_file_system (S3FileSystem): A filesystem object.

        Returns:
            Returns the number of files.
        """

        return len(self.metadata)

    def __getitem__(self, index):
        mel_path = self.metadata_mels[index]
        units_path = self.metadata_units[index]

        mel = np.load(s3_file_system.open(mel_path)).T
        units = np.load(s3_file_system.open(units_path))

        length = 2 * units.shape[0]

        mel = torch.from_numpy(mel[:length, :])
        mel = F.pad(mel, (0, 0, 1, 0))
        units = torch.from_numpy(units)
        return mel, units

    def pad_collate(self, batch):
        mels, units = zip(*batch)

        mels, units = list(mels), list(units)

        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])

        mels = pad_sequence(mels, batch_first=True)
        units = pad_sequence(
            units, batch_first=True, padding_value=100 if self.discrete else 0
        )

        return mels, mels_lengths, units, units_lengths