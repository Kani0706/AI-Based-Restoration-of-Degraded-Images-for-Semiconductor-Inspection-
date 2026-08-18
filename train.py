
import os
import csv
import time
import random
import argparse

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_SEED = 42

DEFAULT_EPOCHS = 10

DEFAULT_BATCH_SIZE = 4

DEFAULT_LEARNING_RATE = 2e-4

L1_WEIGHT = 1.0

SSIM_WEIGHT = 0.2


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN behaviour.
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL IMPORT
# ============================================================

try:

    from models.model import NAFNetSR

except ImportError as exc:

    raise ImportError(
        "\nCould not import NAFNetSR.\n\n"
        "The training script expects the model at:\n"
        "models/model.py\n\n"
        "Repository structure should be:\n"
        "models/\n"
        "    model.py\n"
        "    best_nafnet_sr_l1_ssim_aug.pth\n"
    ) from exc


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Reproducible NAFNet training for "
            "semiconductor image restoration."
        )
    )

    parser.add_argument(
        "--train_noisy",
        required=True,
        help=(
            "Directory containing training "
            "degraded/noisy .npy files."
        )
    )

    parser.add_argument(
        "--train_gt",
        required=True,
        help=(
            "Directory containing training "
            "ground-truth .npy files."
        )
    )

    parser.add_argument(
        "--val_noisy",
        required=True,
        help=(
            "Directory containing validation "
            "degraded/noisy .npy files."
        )
    )

    parser.add_argument(
        "--val_gt",
        required=True,
        help=(
            "Directory containing validation "
            "ground-truth .npy files."
        )
    )

    parser.add_argument(
        "--output_dir",
        default="training_results",
        help=(
            "Directory for checkpoints and "
            "training logs."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of DataLoader workers. "
            "Default is 0 for maximum portability."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed."
    )

    parser.add_argument(
        "--resume",
        default="",
        help=(
            "Optional checkpoint path for "
            "resuming training."
        )
    )

    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help=(
            "Disable paired geometric augmentation."
        )
    )

    return parser.parse_args()


# ============================================================
# DATASET
# ============================================================

class PairedNPYDataset(Dataset):
    """
    Paired degraded/ground-truth NPY dataset.

    Matching filenames must exist in both directories.

    Example:

        noisy/
            000001.npy

        groundtruth/
            000001.npy
    """

    def __init__(
        self,
        noisy_dir,
        gt_dir,
        augmentation=False
    ):

        self.noisy_dir = os.path.abspath(
            noisy_dir
        )

        self.gt_dir = os.path.abspath(
            gt_dir
        )

        self.augmentation = augmentation

        noisy_files = sorted(
            [
                filename
                for filename in os.listdir(
                    self.noisy_dir
                )
                if filename.lower().endswith(
                    ".npy"
                )
            ]
        )

        self.files = []

        missing_gt = []

        for filename in noisy_files:

            gt_path = os.path.join(
                self.gt_dir,
                filename
            )

            if os.path.isfile(gt_path):

                self.files.append(
                    filename
                )

            else:

                missing_gt.append(
                    filename
                )

        if not self.files:

            raise RuntimeError(
                "\nNo paired .npy files found.\n\n"
                f"Noisy directory:\n"
                f"{self.noisy_dir}\n\n"
                f"Ground-truth directory:\n"
                f"{self.gt_dir}\n"
            )

        if missing_gt:

            print(
                f"Warning: {len(missing_gt)} noisy "
                "files do not have matching GT files "
                "and will be skipped."
            )

    def __len__(self):

        return len(self.files)

    def _prepare_array(
        self,
        array,
        filename,
        name
    ):
        """
        Convert NPY array to CHW grayscale format.
        """

        array = np.asarray(
            array,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # H x W
        # ----------------------------------------------------

        if array.ndim == 2:

            array = array[
                None,
                ...
            ]

        # ----------------------------------------------------
        # H x W x 1
        # ----------------------------------------------------

        elif array.ndim == 3:

            if array.shape[-1] == 1:

                array = np.transpose(
                    array,
                    (2, 0, 1)
                )

            # ------------------------------------------------
            # 1 x H x W
            # ------------------------------------------------

            elif array.shape[0] == 1:

                pass

            else:

                raise ValueError(
                    f"{name} image {filename} "
                    f"must be grayscale. "
                    f"Received shape: {array.shape}"
                )

        else:

            raise ValueError(
                f"Invalid {name} image shape "
                f"for {filename}: {array.shape}"
            )

        if array.shape[0] != 1:

            raise ValueError(
                f"{name} image {filename} "
                f"must have one channel. "
                f"Received shape: {array.shape}"
            )

        return array

    def __getitem__(
        self,
        index
    ):

        filename = self.files[index]

        noisy_path = os.path.join(
            self.noisy_dir,
            filename
        )

        gt_path = os.path.join(
            self.gt_dir,
            filename
        )

        # ----------------------------------------------------
        # Load NPY
        # ----------------------------------------------------

        noisy = np.load(
            noisy_path
        )

        gt = np.load(
            gt_path
        )

        # ----------------------------------------------------
        # Convert to CHW grayscale
        # ----------------------------------------------------

        noisy = self._prepare_array(
            noisy,
            filename,
            "noisy"
        )

        gt = self._prepare_array(
            gt,
            filename,
            "ground-truth"
        )

        # ----------------------------------------------------
        # Expected resolution
        # ----------------------------------------------------

        if noisy.shape[1:] != (128, 128):

            raise ValueError(
                f"Expected noisy image "
                f"{filename} to have shape "
                f"(128, 128), but received "
                f"{noisy.shape[1:]}"
            )

        if gt.shape[1:] != (256, 256):

            raise ValueError(
                f"Expected ground-truth image "
                f"{filename} to have shape "
                f"(256, 256), but received "
                f"{gt.shape[1:]}"
            )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        noisy = torch.from_numpy(
            noisy.copy()
        )

        gt = torch.from_numpy(
            gt.copy()
        )

        # ----------------------------------------------------
        # Normalize if required
        #
        # The training pipeline expects values in [0,1].
        # If data is already normalized, no change occurs.
        #
        # ----------------------------------------------------

        noisy = torch.clamp(
            noisy,
            0.0,
            1.0
        )

        gt = torch.clamp(
            gt,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # SAFE PAIRED AUGMENTATION
        #
        # The EXACT same transformation is applied
        # to noisy and ground truth.
        # ----------------------------------------------------

        if self.augmentation:

            # Horizontal flip
            if random.random() < 0.5:

                noisy = torch.flip(
                    noisy,
                    dims=[2]
                )

                gt = torch.flip(
                    gt,
                    dims=[2]
                )

            # Vertical flip
            if random.random() < 0.5:

                noisy = torch.flip(
                    noisy,
                    dims=[1]
                )

                gt = torch.flip(
                    gt,
                    dims=[1]
                )

            # Random 0/90/180/270 degree rotation
            k = random.randint(
                0,
                3
            )

            if k != 0:

                noisy = torch.rot90(
                    noisy,
                    k=k,
                    dims=[1, 2]
                )

                gt = torch.rot90(
                    gt,
                    k=k,
                    dims=[1, 2]
                )

        return noisy, gt


# ============================================================
# SSIM LOSS
# ============================================================

def ssim_loss(
    x,
    y,
    window_size=11,
    sigma=1.5
):
    """
    Differentiable SSIM loss.

    Inputs are expected to be approximately
    normalized to [0,1].

    Returns:
        1 - mean SSIM
    """

    channels = x.size(1)

    coords = torch.arange(
        window_size,
        device=x.device,
        dtype=x.dtype
    )

    coords = (
        coords
        -
        (window_size - 1) / 2.0
    )

    gaussian = torch.exp(
        -(
            coords ** 2
        )
        /
        (
            2.0 * sigma ** 2
        )
    )

    gaussian = (
        gaussian
        /
        gaussian.sum()
    )

    window = (
        gaussian[:, None]
        *
        gaussian[None, :]
    )

    window = (
        window
        .unsqueeze(0)
        .unsqueeze(0)
    )

    window = window.expand(
        channels,
        1,
        window_size,
        window_size
    ).contiguous()

    padding = window_size // 2

    # --------------------------------------------------------
    # Local means
    # --------------------------------------------------------

    mu_x = F.conv2d(
        x,
        window,
        padding=padding,
        groups=channels
    )

    mu_y = F.conv2d(
        y,
        window,
        padding=padding,
        groups=channels
    )

    # --------------------------------------------------------
    # Local statistics
    # --------------------------------------------------------

    mu_x_sq = mu_x * mu_x

    mu_y_sq = mu_y * mu_y

    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(
            x * x,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu_x_sq
    )

    sigma_y_sq = (
        F.conv2d(
            y * y,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu_y_sq
    )

    sigma_xy = (
        F.conv2d(
            x * y,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu_xy
    )

    # --------------------------------------------------------
    # SSIM constants
    # --------------------------------------------------------

    C1 = 0.01 ** 2

    C2 = 0.03 ** 2

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    numerator = (
        (2.0 * mu_xy + C1)
        *
        (2.0 * sigma_xy + C2)
    )

    denominator = (
        (mu_x_sq + mu_y_sq + C1)
        *
        (sigma_x_sq + sigma_y_sq + C2)
    )

    ssim_map = (
        numerator
        /
        (
            denominator
            +
            1e-8
        )
    )

    return (
        1.0
        -
        ssim_map.mean()
    )


# ============================================================
# COMBINED LOSS
# ============================================================

def combined_loss(
    prediction,
    target
):

    # Keep both tensors within the expected range.
    prediction = torch.clamp(
        prediction,
        0.0,
        1.0
    )

    target = torch.clamp(
        target,
        0.0,
        1.0
    )

    # L1 reconstruction loss
    l1 = F.l1_loss(
        prediction,
        target
    )

    # Structural similarity loss
    ssim = ssim_loss(
        prediction,
        target
    )

    total = (
        L1_WEIGHT * l1
        +
        SSIM_WEIGHT * ssim
    )

    return total, l1, ssim


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    prediction = torch.clamp(
        prediction,
        0.0,
        1.0
    )

    target = torch.clamp(
        target,
        0.0,
        1.0
    )

    mse = torch.mean(
        (
            prediction
            -
            target
        ) ** 2
    )

    mse_value = mse.item()

    if mse_value <= 0.0:

        return float("inf")

    return (
        10.0
        *
        np.log10(
            1.0 / mse_value
        )
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_psnr,
    args
):

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_psnr":
            best_psnr,

        "seed":
            args.seed,

        "loss":
            "L1 + 0.2*SSIM",

        "l1_weight":
            L1_WEIGHT,

        "ssim_weight":
            SSIM_WEIGHT,

        "learning_rate":
            args.learning_rate,

        "batch_size":
            args.batch_size,

        "augmentation":
            not args.no_augmentation,

        "model_config": {

            "img_channel":
                1,

            "width":
                32,

            "enc_blocks":
                4,

            "mid_blocks":
                4,

            "dec_blocks":
                4
        }
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Absolute paths
    # --------------------------------------------------------

    train_noisy = os.path.abspath(
        args.train_noisy
    )

    train_gt = os.path.abspath(
        args.train_gt
    )

    val_noisy = os.path.abspath(
        args.val_noisy
    )

    val_gt = os.path.abspath(
        args.val_gt
    )

    output_dir = os.path.abspath(
        args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------

    required_directories = [
        train_noisy,
        train_gt,
        val_noisy,
        val_gt
    ]

    for directory in required_directories:

        if not os.path.isdir(directory):

            raise FileNotFoundError(
                f"\nDirectory not found:\n"
                f"{directory}"
            )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "NAFNET SEMICONDUCTOR IMAGE RESTORATION"
    )
    print(
        "REPRODUCIBLE TRAINING"
    )
    print("=" * 70)

    print(
        "Project directory:",
        PROJECT_DIR
    )

    print(
        "Seed:",
        args.seed
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print(
        "Train noisy:",
        train_noisy
    )

    print(
        "Train GT:",
        train_gt
    )

    print(
        "Validation noisy:",
        val_noisy
    )

    print(
        "Validation GT:",
        val_gt
    )

    print(
        "Output:",
        output_dir
    )

    print()
    print(
        "Epochs:",
        args.epochs
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Learning rate:",
        args.learning_rate
    )

    print(
        "Augmentation:",
        not args.no_augmentation
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = PairedNPYDataset(
        noisy_dir=train_noisy,
        gt_dir=train_gt,
        augmentation=(
            not args.no_augmentation
        )
    )

    val_dataset = PairedNPYDataset(
        noisy_dir=val_noisy,
        gt_dir=val_gt,
        augmentation=False
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = NAFNetSR(
        img_channel=1,
        width=32,
        enc_blocks=4,
        mid_blocks=4,
        dec_blocks=4
    ).to(device)

    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        "Architecture: NAFNetSR"
    )

    print(
        "Model source: models/model.py"
    )

    print(
        "Input channels: 1"
    )

    print(
        "Width: 32"
    )

    print(
        "Encoder blocks: 4"
    )

    print(
        "Middle blocks: 4"
    )

    print(
        "Decoder blocks: 4"
    )

    print()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # --------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------

    start_epoch = 1

    best_psnr = -float("inf")

    if args.resume:

        resume_path = os.path.abspath(
            args.resume
        )

        if not os.path.isfile(
            resume_path
        ):

            raise FileNotFoundError(
                f"\nCheckpoint not found:\n"
                f"{resume_path}"
            )

        print(
            "Resuming training from:"
        )

        print(
            resume_path
        )

        checkpoint = torch.load(
            resume_path,
            map_location=device
        )

        if (
            "model_state_dict"
            not in checkpoint
        ):

            raise ValueError(
                "Invalid checkpoint: "
                "model_state_dict is missing."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        if "epoch" in checkpoint:

            start_epoch = (
                checkpoint["epoch"]
                +
                1
            )

        if "best_psnr" in checkpoint:

            best_psnr = checkpoint[
                "best_psnr"
            ]

        print(
            "Starting epoch:",
            start_epoch
        )

        print(
            "Previous best PSNR:",
            best_psnr
        )

        print()

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    training_start = (
        time.perf_counter()
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs + 1
    ):

        epoch_start = (
            time.perf_counter()
        )

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        total_loss_sum = 0.0

        l1_sum = 0.0

        ssim_sum = 0.0

        for noisy, target in train_loader:

            noisy = noisy.to(
                device,
                non_blocking=True
            )

            target = target.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                noisy
            )

            loss, l1, ssim = (
                combined_loss(
                    prediction,
                    target
                )
            )

            loss.backward()

            optimizer.step()

            total_loss_sum += (
                loss.item()
            )

            l1_sum += (
                l1.item()
            )

            ssim_sum += (
                ssim.item()
            )

        train_loss = (
            total_loss_sum
            /
            len(train_loader)
        )

        train_l1 = (
            l1_sum
            /
            len(train_loader)
        )

        train_ssim = (
            ssim_sum
            /
            len(train_loader)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_sum = 0.0

        psnr_values = []

        with torch.no_grad():

            for noisy, target in val_loader:

                noisy = noisy.to(
                    device,
                    non_blocking=True
                )

                target = target.to(
                    device,
                    non_blocking=True
                )

                prediction = model(
                    noisy
                )

                val_loss, _, _ = (
                    combined_loss(
                        prediction,
                        target
                    )
                )

                val_loss_sum += (
                    val_loss.item()
                )

                # --------------------------------------------
                # PSNR per image
                # --------------------------------------------

                for batch_index in range(
                    prediction.size(0)
                ):

                    psnr = calculate_psnr(
                        prediction[
                            batch_index:
                            batch_index + 1
                        ],
                        target[
                            batch_index:
                            batch_index + 1
                        ]
                    )

                    psnr_values.append(
                        psnr
                    )

        val_loss = (
            val_loss_sum
            /
            len(val_loader)
        )

        val_psnr = float(
            np.mean(
                psnr_values
            )
        )

        epoch_time = (
            time.perf_counter()
            -
            epoch_start
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        is_best = (
            val_psnr
            >
            best_psnr
        )

        if is_best:

            best_psnr = val_psnr

        # ====================================================
        # SAVE LATEST CHECKPOINT
        # ====================================================

        latest_path = os.path.join(
            output_dir,
            "latest_checkpoint.pth"
        )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            best_psnr,
            args
        )

        # ====================================================
        # SAVE BEST CHECKPOINT
        # ====================================================

        if is_best:

            best_path = os.path.join(
                output_dir,
                "best_nafnet_sr_l1_ssim_aug.pth"
            )

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_psnr,
                args
            )

            marker = (
                " <-- BEST MODEL"
            )

        else:

            marker = ""

        # ====================================================
        # HISTORY
        # ====================================================

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_l1":
                    train_l1,

                "train_ssim":
                    train_ssim,

                "val_loss":
                    val_loss,

                "val_psnr":
                    val_psnr,

                "learning_rate":
                    optimizer.param_groups[
                        0
                    ]["lr"],

                "time_seconds":
                    epoch_time
            }
        )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{args.epochs:03d} | "
            f"Train Loss "
            f"{train_loss:.6f} | "
            f"Train L1 "
            f"{train_l1:.6f} | "
            f"Train SSIM "
            f"{train_ssim:.6f} | "
            f"Val Loss "
            f"{val_loss:.6f} | "
            f"Val PSNR "
            f"{val_psnr:.4f} dB | "
            f"{epoch_time:.2f}s"
            f"{marker}"
        )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_path = os.path.join(
        output_dir,
        "training_history.csv"
    )

    with open(
        history_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_l1",
                "train_ssim",
                "val_loss",
                "val_psnr",
                "learning_rate",
                "time_seconds"
            ]
        )

        for row in history:

            writer.writerow(
                [
                    row["epoch"],
                    f"{row['train_loss']:.8f}",
                    f"{row['train_l1']:.8f}",
                    f"{row['train_ssim']:.8f}",
                    f"{row['val_loss']:.8f}",
                    f"{row['val_psnr']:.8f}",
                    f"{row['learning_rate']:.10f}",
                    f"{row['time_seconds']:.4f}"
                ]
            )

    # ========================================================
    # SAVE TRAINING SUMMARY
    # ========================================================

    total_time = (
        time.perf_counter()
        -
        training_start
    )

    summary_path = os.path.join(
        output_dir,
        "training_summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        file.write(
            "NAFNet Semiconductor "
            "Image Restoration\n"
        )

        file.write(
            "========================================\n\n"
        )

        file.write(
            "Method: NAFNet + L1 + 0.2*SSIM\n"
        )

        file.write(
            "Model: models/model.py\n"
        )

        file.write(
            "Input: 128x128 grayscale\n"
        )

        file.write(
            "Target: 256x256 grayscale\n"
        )

        file.write(
            "Augmentation: paired geometric\n"
        )

        file.write(
            "Augmentations: horizontal flip, "
            "vertical flip, "
            "90/180/270 degree rotation\n"
        )

        file.write(
            f"Seed: {args.seed}\n"
        )

        file.write(
            f"Epochs: {args.epochs}\n"
        )

        file.write(
            f"Batch size: {args.batch_size}\n"
        )

        file.write(
            f"Learning rate: "
            f"{args.learning_rate}\n"
        )

        file.write(
            f"Training images: "
            f"{len(train_dataset)}\n"
        )

        file.write(
            f"Validation images: "
            f"{len(val_dataset)}\n"
        )

        file.write(
            f"Best validation PSNR: "
            f"{best_psnr:.6f} dB\n"
        )

        file.write(
            f"Total training time: "
            f"{total_time:.4f} seconds\n"
        )

        file.write(
            f"Device: {device}\n"
        )

        if torch.cuda.is_available():

            file.write(
                "GPU: "
                f"{torch.cuda.get_device_name(0)}\n"
            )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print("=" * 70)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 70)

    print(
        f"Best validation PSNR: "
        f"{best_psnr:.6f} dB"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        os.path.join(
            output_dir,
            "best_nafnet_sr_l1_ssim_aug.pth"
        )
    )

    print()

    print(
        "Training history:"
    )

    print(
        history_path
    )

    print()

    print(
        "Training summary:"
    )

    print(
        summary_path
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()