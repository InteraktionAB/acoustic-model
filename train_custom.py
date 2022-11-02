"""Train the model in distributed fashion

Train the model in distributed fashion

Usage:
    python -m train_custom
"""

import argparse
import json
import os
import typing

import torch

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
    torch.utils.data.distributed.init_process_group(backend=args.backend, init_method="env://", rank=rank, world_size=world_size)
    
    # Clean
    torch.utils.data.distributed.destroy_process_group()


if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser(__doc__)

    # Container arguments
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])) # Instance count
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"]) # Current instance
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    # Find master address
    master_address: str = json.loads(os.environ["SM_TRAINING_ENV"])["master_hostname"]

    # Set master address, port for init_process_group
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = "23456"

    args = parser.parse_args()

    args.backend = "nccl" # Implicit setting backend

    # Start processes
    torch.multiprocessing.spawn(train, nprocs=args.num_gpus, args=(args,))