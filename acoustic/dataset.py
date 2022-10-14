import typing

from pathlib import Path
import numpy as np

import sagemaker
import s3fs
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelDataset(Dataset):
    def __init__(self, root: Path, train: bool = True, discrete: bool = False):
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

        mel = np.load(mel_path).T
        units = np.load(units_path)

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