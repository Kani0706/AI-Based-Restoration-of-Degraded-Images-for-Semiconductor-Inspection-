# AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection

## Overview

This project implements an AI-based image restoration system for degraded semiconductor inspection images using a **NAFNet-based deep learning model**.

The system restores degraded grayscale semiconductor inspection images from **128 × 128** input resolution to **256 × 256** high-quality images. The model is trained using a combination of **L1 loss and SSIM loss**, along with safe paired geometric augmentation to improve reconstruction quality and preserve structural information.

The final solution is packaged for offline evaluation and accepts `.npy` grayscale images as input and generates corresponding restored `.npy` images as output.

---

## Key Features

* NAFNet-based image restoration
* 2× resolution enhancement
* Grayscale semiconductor image processing
* L1 + 0.2 × SSIM loss
* Paired geometric data augmentation
* PyTorch-based deep learning implementation
* Pre-trained model weights included
* `.npy` input and output support
* Same filename maintained between input and output
* Output validation for resolution and value range
* NVIDIA GPU-compatible inference
* Offline inference without external model downloads

---

## Model Summary

| Item               | Value                    |
| ------------------ | ------------------------ |
| Architecture       | NAFNetSR                 |
| Framework          | PyTorch                  |
| Training Method    | Supervised Learning      |
| Input              | 128 × 128 × 1            |
| Output             | 256 × 256 × 1            |
| Scale Factor       | 2×                       |
| Loss Function      | L1 + 0.2 × SSIM          |
| Optimizer          | Adam                     |
| Batch Size         | 4                        |
| Learning Rate      | 0.0002                   |
| Random Seed        | 42                       |
| Input Type         | Grayscale                |
| Target Application | Semiconductor Inspection |

---

## Dataset

The model was trained using paired degraded and ground-truth semiconductor inspection images.

### Dataset Summary

| Dataset    | Number of Images |
| ---------- | ---------------: |
| Training   |            2,560 |
| Validation |              640 |
| Total      |            3,200 |

### Image Properties

* Input resolution: **128 × 128**
* Ground-truth resolution: **256 × 256**
* Image type: **Grayscale**
* File format: **`.npy`**
* Paired images use corresponding filenames.

The training and validation dataset is not included in the final GitHub submission because the evaluation package requires the inference solution and model files rather than the complete training dataset.

---

## Model Architecture

The project uses a NAFNet-based restoration model configured as follows:

* Input channels: 1
* Feature width: 32
* Encoder blocks: 4
* Middle blocks: 4
* Decoder blocks: 4
* Resolution scale: 2×

The model is designed to restore degraded image information while simultaneously producing the required 2× higher spatial resolution.

---

## Loss Function

The model is trained using a combined pixel and structural loss:

```text
Total Loss = L1 Loss + 0.2 × SSIM Loss
```

### L1 Loss

L1 loss minimizes the absolute pixel-level difference between the restored image and the ground-truth image.

### SSIM Loss

SSIM loss helps preserve structural information, edges and local image characteristics.

The combination of L1 and SSIM provides both pixel-level reconstruction and structural preservation.

---

## Data Augmentation

Safe paired geometric augmentation is applied during training.

The following transformations are used:

* Random horizontal flip
* Random vertical flip
* Random 90° rotation
* Random 180° rotation
* Random 270° rotation

The same transformation is applied to both the degraded image and its corresponding ground-truth image to maintain spatial alignment.

---

## Training Configuration

The final model was trained using:

```text
Architecture     : NAFNetSR
Training Images  : 2560
Validation Images: 640
Input Size       : 128 × 128
Target Size      : 256 × 256
Batch Size       : 4
Learning Rate    : 0.0002
Optimizer        : Adam
Loss             : L1 + 0.2 × SSIM
Augmentation     : Enabled
Random Seed      : 42
```

Training and experimentation were performed using a GPU-enabled Google Colab environment.

---

## Repository Structure

```text
AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection/
│
├── run.py
├── requirements.txt
├── README.md
│
└── models/
    ├── model.py
    └── best_nafnet_sr_l1_ssim_aug.pth
```

### File Description

| File / Folder                           | Purpose                                           |
| --------------------------------------- | ------------------------------------------------- |
| `run.py`                                | Main evaluation and inference entry point         |
| `requirements.txt`                      | Required Python dependencies with version details |
| `README.md`                             | Project documentation and execution instructions  |
| `models/model.py`                       | NAFNetSR model architecture                       |
| `models/best_nafnet_sr_l1_ssim_aug.pth` | Trained model weights                             |

---

## Environment Setup

### Requirements

Recommended environment:

* Python 3.10 or 3.11
* PyTorch
* NumPy
* Pillow
* scikit-image
* LPIPS
* tqdm

All required dependencies and their versions are specified in:

```text
requirements.txt
```

### Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Enter the repository:

```bash
cd AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU execution, the installed PyTorch package must have CUDA support compatible with the target GPU.

---

## Evaluation / Inference

The official evaluation entry point is:

```text
run.py
```

The solution is executed using:

```bash
python run.py <input-dir> <output-dir>
```

### Example

```bash
python run.py test/noisy test_output
```

Windows PowerShell:

```powershell
python run.py "C:\path\to\test\noisy" "C:\path\to\test_output"
```

The script requires no user interaction or manual configuration during execution.

---

## Input Contract

`run.py` reads all `.npy` files from the specified input directory.

Example:

```text
input/
├── 000000.npy
├── 000001.npy
├── 000002.npy
└── ...
```

### Input Requirements

* File format: `.npy`
* Image type: grayscale
* Expected resolution: **128 × 128**
* Numerical image array
* One image per `.npy` file

The input filenames are preserved for the corresponding output files.

---

## Output Contract

For every input `.npy` file, the system generates exactly one restored `.npy` file.

Example:

```text
input/
├── 000000.npy
├── 000001.npy
└── 000002.npy

output/
├── 000000.npy
├── 000001.npy
└── 000002.npy
```

### Output Requirements

* File format: `.npy`
* Grayscale array
* Target resolution: **256 × 256**
* Values within **[0, 1]**
* No `NaN` values
* No `Inf` values
* Same filename as the input
* One output for every input

The output directory is automatically created if it does not already exist.

---

## Model Weights

The trained model checkpoint is included in:

```text
models/best_nafnet_sr_l1_ssim_aug.pth
```

The corresponding model architecture is:

```text
models/model.py
```

All required model files are included in the repository so that inference does not require downloading additional weights.

---

## Validation Results

The final model was evaluated on **640 validation images**.

| Metric        |           Result |
| ------------- | ---------------: |
| Average PSNR  | **28.388865 dB** |
| Average SSIM  |     **0.772883** |
| Average LPIPS |     **0.261666** |

### Metric Interpretation

**PSNR — 28.388865 dB**

Measures pixel-level reconstruction quality. Higher PSNR generally indicates lower reconstruction error.

**SSIM — 0.772883**

Measures structural similarity between the restored image and ground truth. Higher SSIM generally indicates better structural preservation.

**LPIPS — 0.261666**

Measures perceptual similarity. Lower LPIPS generally indicates greater perceptual similarity.

---

## Baseline vs Augmentation

The project evaluated the effect of paired geometric augmentation by comparing:

* NAFNet baseline without augmentation
* NAFNet with paired geometric augmentation

Both configurations were evaluated using the same validation dataset.

The comparison was performed using:

* PSNR
* SSIM
* LPIPS

The final selected model is the **NAFNet + L1 + 0.2 × SSIM + augmentation** configuration.

---

## Test Processing

The inference pipeline was tested on **400 test images**, producing a corresponding restored output for each input image.

```text
Test inputs     : 400
Restored outputs: 400
Input resolution: 128 × 128
Output resolution: 256 × 256
Batch size      : 4
```

Inference throughput is hardware-dependent and may vary depending on the GPU and execution environment.

---

## Hardware and Software

### Hardware

* NVIDIA GPU environment
* Google Colab GPU used for training and experimentation

### Software

* Python
* PyTorch
* NumPy
* scikit-image
* LPIPS
* Matplotlib
* tqdm
* Google Colab

---

## Offline Evaluation

The final submission is designed to operate without internet connectivity.

The evaluation system does not require:

* Internet access
* API keys
* External model downloads
* User interaction
* Manual configuration

All required model architecture and trained weights are included in the `models/` directory.

---

## Assumptions

The solution assumes:

1. The input directory contains valid `.npy` grayscale images.
2. The evaluation inputs follow the expected 128 × 128 resolution.
3. The restoration task requires 2× spatial resolution enhancement.
4. The output must be a grayscale `.npy` array.
5. Output values must remain within `[0, 1]`.
6. Each input file must produce exactly one output file.
7. Input filenames must be preserved in the output directory.

---

## Limitations

* The model is designed for grayscale semiconductor inspection images.
* The expected input resolution is 128 × 128.
* The target output resolution is 256 × 256.
* Performance may vary for degradation patterns significantly different from those represented in the training dataset.
* Reported PSNR, SSIM and LPIPS values are specific to the project validation dataset.
* Inference speed depends on the available hardware.

---

## External Resource Disclosure

The project uses PyTorch and a NAFNet-based architecture for deep-learning image restoration.

The final repository contains the model architecture and trained weights required for inference.

No external API, internet-based inference service, or additional model download is required during evaluation.

---

## Team Members

**Team Name:** Wafer Endeavours

**Institution:** Chennai Institute of Technology

* S Kanimozhi
* M Pavithra
* R Pavithra
* S Kanagalakshmi

---

## Final Execution

The evaluator can run the submitted solution using:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py test/noisy test_output
```

The system reads all `.npy` files from the input directory and generates one restored `.npy` file for each input while preserving filenames and satisfying the required output format.
