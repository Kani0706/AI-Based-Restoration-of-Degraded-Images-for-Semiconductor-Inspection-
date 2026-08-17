# ============================================================
# NAFNet Semiconductor Image Restoration
# FINAL EVALUATION ENTRY SCRIPT
#
# Usage:
#
#     python run.py <input-dir> <output-dir>
#
# Example:
#
#     python run.py "test/noisy" "submission_output"
#
# Requirements:
# - Reads all .npy files from input directory
# - Creates output directory automatically
# - Produces exactly one .npy output per input
# - Preserves input filenames
# - Produces grayscale (H, W) outputs
# - Produces 2x restored resolution
# - Output values are guaranteed to be in [0, 1]
# - Rejects NaN/Inf values
# - Supports NVIDIA CUDA GPU when available
# - Does not require internet or additional downloads
# - Uses model and weights bundled in models/
# ============================================================

import os
import sys
import argparse

import numpy as np
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODELS_DIR,
    "model.py"
)

CHECKPOINT_FILE = os.path.join(
    MODELS_DIR,
    "best_nafnet_sr_l1_ssim_aug.pth"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.isfile(MODEL_FILE):

    raise FileNotFoundError(
        "Required model file was not found:\n"
        f"{MODEL_FILE}"
    )

if not os.path.isfile(CHECKPOINT_FILE):

    raise FileNotFoundError(
        "Required model checkpoint was not found:\n"
        f"{CHECKPOINT_FILE}"
    )


# ============================================================
# IMPORT MODEL
# ============================================================

if MODELS_DIR not in sys.path:

    sys.path.insert(
        0,
        MODELS_DIR
    )

try:

    from model import NAFNetSR

except ImportError as error:

    raise ImportError(
        "Could not import NAFNetSR from models/model.py.\n"
        f"Original error: {error}"
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "NAFNet-based semiconductor image restoration "
        "for evaluation."
    )
)

parser.add_argument(
    "input_dir",
    type=str,
    help="Directory containing degraded .npy images."
)

parser.add_argument(
    "output_dir",
    type=str,
    help="Directory where restored .npy images will be saved."
)

args = parser.parse_args()


# ============================================================
# ABSOLUTE PATHS
# ============================================================

INPUT_DIR = os.path.abspath(
    args.input_dir
)

OUTPUT_DIR = os.path.abspath(
    args.output_dir
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NAFNET SEMICONDUCTOR IMAGE RESTORATION")
print("EVALUATION INFERENCE")
print("=" * 70)

print(
    f"Device       : {device}"
)

print(
    f"Input        : {INPUT_DIR}"
)

print(
    f"Output       : {OUTPUT_DIR}"
)

print(
    f"Model        : {MODEL_FILE}"
)

print(
    f"Checkpoint   : {CHECKPOINT_FILE}"
)


# ============================================================
# VALIDATE INPUT DIRECTORY
# ============================================================

if not os.path.isdir(INPUT_DIR):

    raise FileNotFoundError(
        "Input directory does not exist:\n"
        f"{INPUT_DIR}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND ALL NPY INPUT FILES
# ============================================================

input_files = sorted(
    filename
    for filename in os.listdir(INPUT_DIR)
    if filename.lower().endswith(".npy")
)


if len(input_files) == 0:

    raise RuntimeError(
        "No .npy files were found in the input directory:\n"
        f"{INPUT_DIR}"
    )


print()
print(
    f"Input images : {len(input_files)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading NAFNet model...")


model = NAFNetSR(
    img_channel=1,
    width=32,
    enc_blocks=4,
    mid_blocks=4,
    dec_blocks=4
).to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device
)


# ============================================================
# SUPPORT CHECKPOINT FORMAT
# ============================================================

if (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "N/A"
    )

    checkpoint_psnr = checkpoint.get(
        "best_psnr",
        "N/A"
    )

else:

    state_dict = checkpoint

    checkpoint_epoch = "N/A"

    checkpoint_psnr = "N/A"


# ============================================================
# REMOVE DataParallel PREFIX IF PRESENT
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[
            len("module.") :
        ]

    clean_state_dict[key] = value


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.eval()


print(
    "Model loaded successfully."
)

print(
    f"Checkpoint epoch : {checkpoint_epoch}"
)

print(
    f"Best PSNR        : {checkpoint_psnr}"
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Parameters       : {parameter_count:,}"
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("STARTING INFERENCE")
print("=" * 70)


processed = 0


with torch.inference_mode():

    for filename in input_files:

        # ====================================================
        # INPUT PATH
        # ====================================================

        input_path = os.path.join(
            INPUT_DIR,
            filename
        )


        # ====================================================
        # OUTPUT PATH
        # ====================================================

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )


        # ====================================================
        # LOAD INPUT
        # ====================================================

        image = np.load(
            input_path
        )


        # ====================================================
        # CONVERT TO FLOAT32
        # ====================================================

        image = image.astype(
            np.float32,
            copy=False
        )


        # ====================================================
        # VALIDATE INPUT
        # ====================================================

        if image.ndim != 2:

            raise ValueError(
                "\nInvalid input image.\n"
                f"File: {filename}\n"
                f"Shape: {image.shape}\n"
                "Expected grayscale array with shape (H, W)."
            )


        if image.shape[0] == 0 or image.shape[1] == 0:

            raise ValueError(
                "\nInput image has invalid dimensions.\n"
                f"File: {filename}\n"
                f"Shape: {image.shape}"
            )


        # ====================================================
        # CHECK INPUT FOR NaN / INF
        # ====================================================

        if not np.isfinite(image).all():

            raise ValueError(
                "\nInput contains NaN or Inf values.\n"
                f"File: {filename}"
            )


        # ====================================================
        # SAVE ORIGINAL INPUT SIZE
        # ====================================================

        input_height = image.shape[0]
        input_width = image.shape[1]


        # ====================================================
        # NUMPY → TORCH
        #
        # Shape:
        #
        # (H, W)
        #     ↓
        # (1, H, W)
        #     ↓
        # (1, 1, H, W)
        # ====================================================

        input_tensor = torch.from_numpy(
            image
        ).unsqueeze(
            0
        ).unsqueeze(
            0
        )


        # ====================================================
        # CPU → GPU / CPU
        # ====================================================

        input_tensor = input_tensor.to(
            device
        )


        # ====================================================
        # MODEL INFERENCE
        # ====================================================

        output_tensor = model(
            input_tensor
        )


        # ====================================================
        # EXPECTED TARGET RESOLUTION
        #
        # The trained model performs 2× restoration:
        #
        # 128 × 128 → 256 × 256
        # ====================================================

        expected_height = (
            input_height * 2
        )

        expected_width = (
            input_width * 2
        )


        # ====================================================
        # VALIDATE MODEL OUTPUT DIMENSIONS
        # ====================================================

        if output_tensor.ndim != 4:

            raise RuntimeError(
                "\nUnexpected model output dimensions.\n"
                f"File: {filename}\n"
                f"Output shape: "
                f"{tuple(output_tensor.shape)}"
            )


        if output_tensor.shape[0] != 1:

            raise RuntimeError(
                "\nUnexpected batch dimension.\n"
                f"Output shape: "
                f"{tuple(output_tensor.shape)}"
            )


        if output_tensor.shape[1] != 1:

            raise RuntimeError(
                "\nUnexpected number of output channels.\n"
                f"Output shape: "
                f"{tuple(output_tensor.shape)}\n"
                "Expected one grayscale channel."
            )


        if (
            output_tensor.shape[2]
            != expected_height
            or
            output_tensor.shape[3]
            != expected_width
        ):

            raise RuntimeError(
                "\nIncorrect restored resolution.\n"
                f"File: {filename}\n"
                f"Input resolution: "
                f"{input_height} x {input_width}\n"
                f"Output resolution: "
                f"{output_tensor.shape[2]} x "
                f"{output_tensor.shape[3]}\n"
                f"Expected resolution: "
                f"{expected_height} x "
                f"{expected_width}"
            )


        # ====================================================
        # CHECK MODEL OUTPUT FOR NaN / INF
        # ====================================================

        if not torch.isfinite(
            output_tensor
        ).all():

            raise RuntimeError(
                "\nModel produced NaN or Inf values.\n"
                f"File: {filename}"
            )


        # ====================================================
        # CLIP OUTPUT TO [0,1]
        # ====================================================

        output_tensor = torch.clamp(
            output_tensor,
            min=0.0,
            max=1.0
        )


        # ====================================================
        # TORCH → NUMPY
        # ====================================================

        restored = (
            output_tensor
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False
            )
        )


        # ====================================================
        # FINAL OUTPUT VALIDATION
        # ====================================================

        if restored.ndim != 2:

            raise RuntimeError(
                "\nRestored image is not grayscale 2-D.\n"
                f"File: {filename}\n"
                f"Output shape: {restored.shape}"
            )


        if restored.shape != (
            expected_height,
            expected_width
        ):

            raise RuntimeError(
                "\nFinal output has incorrect resolution.\n"
                f"File: {filename}\n"
                f"Output shape: {restored.shape}\n"
                f"Expected: "
                f"({expected_height}, {expected_width})"
            )


        if not np.isfinite(
            restored
        ).all():

            raise RuntimeError(
                "\nFinal output contains NaN or Inf.\n"
                f"File: {filename}"
            )


        # ====================================================
        # FINAL RANGE ENFORCEMENT
        # ====================================================

        restored = np.clip(
            restored,
            0.0,
            1.0
        ).astype(
            np.float32,
            copy=False
        )


        # ====================================================
        # FINAL RANGE CHECK
        # ====================================================

        if (
            restored.min() < 0.0
            or
            restored.max() > 1.0
        ):

            raise RuntimeError(
                "\nOutput values are outside [0,1].\n"
                f"File: {filename}"
            )


        # ====================================================
        # SAVE .NPY
        #
        # IMPORTANT:
        # Same filename as input.
        #
        # Example:
        #
        # input: 000000.npy
        # output: 000000.npy
        # ====================================================

        np.save(
            output_path,
            restored
        )


        # ====================================================
        # VERIFY SAVED FILE
        # ====================================================

        if not os.path.isfile(
            output_path
        ):

            raise RuntimeError(
                "\nOutput file was not created.\n"
                f"File: {output_path}"
            )


        processed += 1


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            processed == 1
            or
            processed % 10 == 0
            or
            processed == len(input_files)
        ):

            print(
                f"Processed "
                f"{processed}/{len(input_files)}"
            )


# ============================================================
# FINAL OUTPUT COUNT
# ============================================================

output_files = sorted(
    filename
    for filename in os.listdir(OUTPUT_DIR)
    if filename.lower().endswith(".npy")
)


# ============================================================
# VERIFY OUTPUT COUNT
# ============================================================

if len(output_files) != len(input_files):

    raise RuntimeError(
        "\nOutput count does not match input count.\n"
        f"Input files : {len(input_files)}\n"
        f"Output files: {len(output_files)}"
    )


# ============================================================
# VERIFY FILENAMES
# ============================================================

input_names = set(
    input_files
)

output_names = set(
    output_files
)


if input_names != output_names:

    missing = sorted(
        input_names - output_names
    )

    unexpected = sorted(
        output_names - input_names
    )

    raise RuntimeError(
        "\nInput/output filenames do not match.\n"
        f"Missing outputs: {missing[:10]}\n"
        f"Unexpected outputs: {unexpected[:10]}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)

print(
    f"Images processed : {processed}"
)

print(
    f"Outputs created  : {len(output_files)}"
)

print(
    f"Output directory : {OUTPUT_DIR}"
)

print(
    f"Device           : {device}"
)

print()
print(
    "All outputs passed validation:"
)

print(
    "  ✓ One .npy output per input"
)

print(
    "  ✓ Same filenames"
)

print(
    "  ✓ Grayscale (H, W)"
)

print(
    "  ✓ 2× target resolution"
)

print(
    "  ✓ Values within [0, 1]"
)

print(
    "  ✓ No NaN / Inf values"
)

print("=" * 70)
print("Evaluation inference finished successfully.")
print("=" * 70)