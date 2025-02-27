import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from acoustic import AcousticModel
from acoustic.dataset import MelDataset, MelDatasetOS
from acoustic.utils import Metric, save_checkpoint, load_checkpoint, plot_spectrogram


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

########################################################################################
# Define hyperparameters for training:
########################################################################################

BATCH_SIZE = 64
LEARNING_RATE = 4e-4
BETAS = (0.8, 0.99)
WEIGHT_DECAY = 1e-5
STEPS = 80000
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 1000
CHECKPOINT_INTERVAL = 1000
BACKEND = "nccl"
INIT_METHOD = "tcp://127.0.0.1:54321"

print(json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname'])

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

#os.environ["MASTER_ADDR"] = "127.0.0.1"
#os.environ["MASTER_PORT"] = find_free_port()


def train(args):
    rank=0
    world_size=4
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD,
    )

    ####################################################################################
    # Setup logging utilities:
    ####################################################################################

    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir / f"{args.checkpoint_dir.stem}.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    ####################################################################################
    # Initialize models and optimizer
    ####################################################################################

    acoustic = AcousticModel().to(rank)

    acoustic = DDP(acoustic, device_ids=[rank])

    optimizer = optim.AdamW(
        acoustic.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    ####################################################################################
    # Initialize datasets and dataloaders
    ####################################################################################

    train_dataset = MelDatasetOS(
        root=args.dataset_dir,
        train=True,
        discrete=args.discrete,
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = dataloader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=train_dataset.pad_collate,
        num_workers=8,
        pin_memory=true,
        shuffle=false,
        drop_last=true,
    )

    validation_dataset = MelDataset(
        root=args.dataset_dir,
        train=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    ####################################################################################
    # Load checkpoint if args.resume is set
    ####################################################################################

    if args.resume is not None:
        global_step, best_loss = load_checkpoint(
            load_path=args.resume,
            acoustic=acoustic,
            optimizer=optimizer,
            rank=rank,
            logger=logger,
        )
    else:
        global_step, best_loss = 0, float("inf")

    # =================================================================================#
    # Start training loop
    # =================================================================================#

    n_epochs = STEPS // len(train_loader) + 1
    start_epoch = global_step // len(train_loader) + 1

    logger.info("**" * 40)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"# of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info("**" * 40 + "\n")

    average_loss = Metric()
    epoch_loss = Metric()

    validation_loss = Metric()

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)

        acoustic.train()
        epoch_loss.reset()

        for mels, mels_lengths, units, units_lengths in train_loader:
            mels, mels_lengths = mels.to(rank), mels_lengths.to(rank)
            units, units_lengths = units.to(rank), units_lengths.to(rank)

            ############################################################################
            # Compute training loss
            ############################################################################

            optimizer.zero_grad()

            mels_ = acoustic(units, mels[:, :-1, :])

            loss = F.l1_loss(mels_, mels[:, 1:, :], reduction="none")
            loss = torch.sum(loss, dim=(1, 2)) / (mels_.size(-1) * mels_lengths)
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

            global_step += 1

            ############################################################################
            # Update and log training metrics
            ############################################################################

            average_loss.update(loss.item())
            epoch_loss.update(loss.item())

            if rank == 0 and global_step % LOG_INTERVAL == 0:
                writer.add_scalar(
                    "train/loss",
                    average_loss.value,
                    global_step,
                )
                average_loss.reset()

            # --------------------------------------------------------------------------#
            # Start validation loop
            # --------------------------------------------------------------------------#

            if global_step % VALIDATION_INTERVAL == 0:
                acoustic.eval()
                validation_loss.reset()

                for i, (mels, units) in enumerate(validation_loader, 1):
                    mels, units = mels.to(rank), units.to(rank)

                    with torch.no_grad():
                        mels_ = acoustic(units, mels[:, :-1, :])
                        loss = F.l1_loss(mels_, mels[:, 1:, :])

                    ####################################################################
                    # Update validation metrics and log generated mels
                    ####################################################################

                    validation_loss.update(loss.item())

                    if rank == 0:
                        writer.add_figure(
                            f"generated/mel_{i}",
                            plot_spectrogram(
                                mels_.squeeze().transpose(0, 1).cpu().numpy()
                            ),
                            global_step,
                        )

                acoustic.train()

                ############################################################################
                # Log validation metrics
                ############################

                if rank == 0:
                    writer.add_scalar(
                        "validation/loss",
                        validation_loss.value,
                        global_step,
                    )
                    logger.info(
                        f"valid -- epoch: {epoch}, loss: {validation_loss.value:.4f}"
                    )

                new_best = best_loss > validation_loss.value
                if new_best or global_step % CHECKPOINT_INTERVAL:
                    if new_best:
                        logger.info("-------- new best model found!")
                        best_loss = validation_loss.value

                    if rank == 0:
                        save_checkpoint(
                            checkpoint_dir=args.checkpoint_dir,
                            acoustic=acoustic,
                            optimizer=optimizer,
                            step=global_step,
                            loss=validation_loss.value,
                            best=new_best,
                            logger=logger,
                        )

            # -----------------------------------------------------------------------------#
            # End validation loop
            # -----------------------------------------------------------------------------#

        ####################################################################################
        # Log training metrics
        ####################################################################################

        logger.info(f"train -- epoch: {epoch}, loss: {epoch_loss.value:.4f}")

        # =================================================================================#
        # End training loop
        # ==================================================================================#

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the acoustic model.")
    parser.add_argument(
        "--dataset_dir",
        metavar="dataset-dir",
        help="path to the data directory.",
        type=Path,
        required=False,
        default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument(
        "--checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory.",
        required=False,
        default=Path(os.environ["SM_MODEL_DIR"]),
        type=Path,
    )
    parser.add_argument(
        "--resume",
        help="path to the checkpoint to resume from.",
        type=Path,
    )
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="use discrete units.",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    # mp.spawn(
    #     train,
    #     args=(world_size, args),
    #     nprocs=world_size,
    #     join=True,
    # )

    train(args)

    
