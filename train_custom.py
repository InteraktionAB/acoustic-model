"""Train the model in distributed fashion

Train the model in distributed fashion

Usage:
    python -m train_custom
"""

import argparse
import os
import typing

import torch

def train(
    rank: int, 
    arguments: typing.Tuple
    ) -> None:

    """Train the model

    Train the model in multiple GPUs.

    Args:
        rank: The id of the process.
        arguments: Arguments to the funtion.

    Returns:
        None
    """

    arguments.initialize_distribution(
        backend=arguments.backend,
        rank=rank,
        world_size=arguments.world_size,
    )

if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser(__doc__)

    parser.add_argument(
        "--dataset", 
        type=str,
        help="Root of the dataset",
        required=False,
        default=os.environ["SM_CHANNEL_TRAINING"],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Location to store checkpoint",
        required=False,
        default=os.environ["SM_MODEL_DIR"],
    )

    parser.add_argument(
        "--hosts",
        type=list,
        help="List of hosts to use",
        required=False,
        default=[0, 1, 2, 3],
    )

    parser.add_argument(
        "--backend",
        type=str,
        help="Backend for distributed training",
        required=False,
        default="nccl"
    )

    arguments = parser.parse_args()

    os.environ["WORLD_SIZE"] = str(arguments.hosts)

    arguments.initialize_distribution = torch.distributed.init_process_group

    torch.multiprocessing.spawn(
        fn=train,
        args=(arguments,),
        nprocs=len(arguments.hosts),
        join=True,
        )
