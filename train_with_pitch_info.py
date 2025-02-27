"""Train with pitch info.

Train acoustic decoder conditioned on
pitch.

Usage:
    python -m train_with_info

"""

from pathlib import Path
import argparse
import json
import logging
import os
import typing

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch

from acoustic import LinearNetwork, AcousticModelWithPitch, AcousticModel
from acoustic.dataset import MelAndPitchDataset

def train(
    gpu: int, 
    args: typing.Tuple
    ) -> None:

    """Train the model

    Train the model in multiple GPUs.

    Args:
        gpu: The id of the current gpu.
        args: Arguments to the funtion.

    Returns:
        None

    """
    world_size: int = len(args.hosts) * args.num_gpus # Total Num. GPUs
    rank: int = args.hosts.index(args.current_host) * args.num_gpus + gpu # Current process is GPU ID in the current host

    # Set them environ
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initiate process
    dist.init_process_group(backend=args.backend, init_method="env://", rank=rank, world_size=world_size)

    # Linear network
    linear_network: LinearNetwork = LinearNetwork(input_size=2, output_size=512,)

    # Define the model
    acoustic_model: AcousticModel = AcousticModel()

    # Acoustic model with pitch
    acoustic_model_with_pitch: AcousticModelWithPitch = AcousticModelWithPitch(
        acoustic_model, linear_network,)

    # Set device
    acoustic_model_with_pitch.to(gpu)

    # Set device id
    acoustic: DDP = DDP(acoustic_model_with_pitch, device_ids=[gpu])

    # Optimizer
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(acoustic.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay,)

    # Train dataset
    train_dataset: torch.nn.data.Dataset = MelAndPitchDataset(root=args.dataset_dir, train=True, discrete=args.discrete)

    # Data sampler
    train_sampler: DistributedSampler = DistributedSampler(train_dataset, drop_last=True,)

    # Train data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=train_dataset.pad_collate,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    # Number of epochs
    num_of_epochs: int = args.steps // len(train_loader) + 1

    for epoch in range(num_of_epochs):
        train_sampler.set_epoch(epoch)
        acoustic_model.train()
        for mels, mels_lengths, units, units_lengths, pitches, _ in train_loader:
            mels, mels_lengths = mels.to(rank), mels_lengths.to(rank)
            units, units_lengths = units.to(rank), units_lengths.to(rank)
            pitches, _ = pitches.to(rank), _

            optimizer.zero_grad()

            mels_ = acoustic_model_with_pitch(pitches, units, mels[:, :-1, :])

            loss = F.l1_loss(mels_, mels[:, 1:, :], reduction="none")
            loss = torch.sum(loss, dim=(1, 2)) / (mels_.size(-1) * mels_lengths)
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

        args.logger.info(f"loss: {loss}") 

    # Clean
    dist.destroy_process_group()


if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser(__doc__)

    # Container arguments
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])) # Instance count
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"]) # Current instance
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    # Train arguments
    parser.add_argument(
        "--dataset_dir",
        metavar="dataset-dir",
        help="path to the data directory.",
        type=Path,
        required=False,
        default=os.environ["SM_CHANNEL_TRAINING"],
    )

    # Find master address
    master_address: str = json.loads(os.environ["SM_TRAINING_ENV"])["master_hostname"]

    # Set master address, port for init_process_group
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = "23456"

    args = parser.parse_args()

    args.backend = "nccl" # Implicit setting backend
    args.lr: float = 4e-4 # Implicit setting learning rate
    args.betas: tuple = (0.8, 0.99) # Implicit setting betas
    args.weight_decay: float = 1e-5 # Implicit setting weight decay
    args.discrete: bool = False # Implicit setting discrete
    args.batch_size: int = 5 # Implicit batch setting
    args.steps: int = 10 # Implicit setting of steps

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    args.logger = logger

    # Start processes
    mp.spawn(train, nprocs=args.num_gpus, args=(args,))