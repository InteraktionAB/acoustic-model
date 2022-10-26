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
        default=json.loads(os.environ["SM_HOSTS"]),
    )

    parser.add_argument(
        "--backend",
        type=str,
        help="Backend for distributed training",
        required=False,
        default="nccl"
    )

    parser.add_argument(
        "--current_host",
        type=str,
        help="The current host",
        required=False,
        default=os.environ["SM_CURRENT_HOST"],
    )

    arguments = parser.parse_args()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9999"

    os.environ["WORLD_SIZE"] = str(len(arguments.hosts))
    os.environ["RANK"] = str(arguments.hosts.index(arguments.current_host))

    arguments.initialize_distribution = torch.distributed.init_process_group

    arguments.initialize_distribution(
        backend=arguments.backend,
        rank=arguments.hosts.index(arguments.current_host),
        world_size=len(arguments.hosts)
    )