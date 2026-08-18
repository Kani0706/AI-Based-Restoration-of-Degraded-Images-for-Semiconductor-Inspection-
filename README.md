# AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection

## Overview

This project implements an AI-based image restoration system for degraded semiconductor inspection images. The solution uses a **NAFNet-based deep learning model** to restore degraded low-resolution grayscale images from **128 × 128** to the required **256 × 256** target resolution.

The model is trained using a combination of **L1 loss and SSIM loss**, with safe paired geometric augmentation to improve restoration performance while preserving structural information. The final solution is packaged for offline evaluation and processes `.npy` input images to generate corresponding restored `.npy` outputs.

The inference pipeline is designed to run on an NVIDIA GPU without requiring internet access, API keys, additional model downloads, or manual configuration.

---

## Key Features

* NAFNet-based image restoration architecture
* 2× resolution restoration from 128 × 128 to 256 × 256
* Grayscale semiconductor inspection image processing
* L1 + 0.2 × SSIM combined loss
* Safe paired geometric augmentation
* Horizontal flip, vertical flip and 90° rotation augmentation
* PyTorch-based training and inference
* Pre-trained model weights included in the repository
* `.npy` input and output support
* One-to-one input/output filename preservation
* Output validation for correct resolution and value range
* Offline NVIDIA GPU-compatible inference
* Reproducible training using a fixed random seed

---

## Model Summary

| Item                  | Value                           |
| --------------------- | ------------------------------- |
| Architecture          | NAFNetSR                        |
| Training Approach     | Supervised Deep Learning        |
| Framework             | PyTorch                         |
| Input                 | 128 × 128 × 1                   |
| Output                | 256 × 256 × 1                   |
| Scale Factor          | 2×                              |
| Loss Function         | L1 + 0.2 × SSIM                 |
| Optimizer             | Adam                            |
| Default Batch Size    | 4                               |
| Default Learning Rate | 0.0002                          |
| Random Seed           | 42                              |
| Base Width            | 32                              |
| Encoder Blocks        | 4                               |
| Middle Blocks         | 4                               |
| Decoder Blocks        | 4                               |
| Target                | Semiconductor Image Restoration |

---

## Dataset Summary

The project uses paired degraded and ground-truth semiconductor inspection images.

| Dataset    | Images |
| ---------- | -----: |
| Training   |  2,560 |
| Validation |    640 |
| Total      |  3,200 |

### Dataset Characteristics

* Input images: degraded/noisy low-resolution images
* Ground truth: high-resolution reference images
* Input resolution: **128 × 128**
* Ground-truth resolution: **256 × 256**
* Image type: grayscale
* Storage format: `.npy`
* Training/validation pairing: matching filenames

Example:

```text
train/
├── noisy/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
│
└── groundtruth/
    ├── 000000.npy
    ├── 000001.npy
    └── ...
```

The complete training dataset is not required for final inference and is therefore not included in the GitHub repository.

---

## Training Method

The model is trained using the following objective:

```text
Total Loss = L1 Loss + 0.2 × SSIM Loss
```

### L1 Loss

L1 loss minimizes the absolute pixel difference between the restored image and the ground-truth image.

### SSIM Loss

SSIM loss encourages preservation of structural information, edges and local image characteristics.

### Data Augmentation

Training uses safe paired geometric augmentation:

* Horizontal flip
* Vertical flip
* 90° rotation
* 180° rotation
* 270° rotation

The exact same transformation is applied to the degraded image and its corresponding ground-truth image to maintain pixel-level alignment.

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

| File / Folder                           | Purpose                                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------------------------- |
| `run.py`                                | Main evaluation and inference entry point                                                 |
| `requirements.txt`                      | Required Python dependencies with version details                                         |
| `README.md`                             | Project description, setup, execution instructions, input/output requirements and results |
| `models/model.py`                       | NAFNetSR model architecture                                                               |
| `models/best_nafnet_sr_l1_ssim_aug.pth` | Trained NAFNet model weights                                                              |

The complete submission is self-contained for inference. The evaluator does not need the training dataset, training scripts, Google Colab notebooks, or external model downloads.


---

## Environment Setup

### Requirements

Recommended environment:

```text
Python 3.10 / 3.11
PyTorch
NumPy
Pillow
scikit-image
LPIPS
tqdm
```

All required Python dependencies and their versions are listed in:

```text
requirements.txt
```
## Final Evaluation / Inference

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

Windows PowerShell example:

```powershell
py run.py "C:\path\to\test\noisy" "C:\path\to\test_output"
```

The script does not require interactive input.

---

## Input Requirements

The input directory must contain degraded `.npy` images.

Example:

```text
input/
├── 000000.npy
├── 000001.npy
├── 000002.npy
└── ...
```

### Input Contract

* File format: `.npy`
* Image type: grayscale
* Expected input resolution: **128 × 128**
* One image per `.npy` file
* Numerical image array
* Input filenames are preserved in the output

---

## Output Requirements

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

The output satisfies the following requirements:

* `.npy` format
* Grayscale array
* Target resolution: **256 × 256**
* Output values within **[0, 1]**
* No `NaN` values
* No `Inf` values
* Same filename as the corresponding input
* One output for every input

The output directory is automatically created if it does not already exist.

---

## Trained Model

The trained model checkpoint is included in:

```text
weights/best_nafnet_sr_l1_ssim_aug.pth
```

The model does not require downloading weights from an external server during evaluation.

The supporting model implementation is available in:

```text
models/model.py
```

---

## Training Configuration

The final training configuration used for the reported model is:

```text
Architecture    : NAFNetSR
Input            : 128 × 128
Ground Truth     : 256 × 256
Training Images  : 2560
Validation Images: 640
Batch Size       : 4
Learning Rate    : 0.0002
Optimizer        : Adam
Loss             : L1 + 0.2 × SSIM
Augmentation     : Enabled
Random Seed      : 42
```

Training was performed using a GPU-enabled environment through Google Colab.

---

## Results

The final NAFNet model with **L1 + SSIM loss and paired augmentation** was evaluated on **640 validation images**.

| Metric        |           Result |
| ------------- | ---------------: |
| Average PSNR  | **28.388865 dB** |
| Average SSIM  |     **0.772883** |
| Average LPIPS |     **0.261666** |

### Metric Interpretation

**PSNR – 28.388865 dB**

Measures pixel-level reconstruction quality. Higher PSNR generally indicates lower reconstruction error.

**SSIM – 0.772883**

Measures structural similarity between the restored and ground-truth images. Higher values indicate better structural preservation.

**LPIPS – 0.261666**

Measures perceptual similarity. Lower values generally indicate greater perceptual similarity.

---

## Baseline vs Augmentation

The project also evaluates the effect of paired geometric augmentation by comparing:

1. NAFNet baseline without augmentation
2. NAFNet with paired geometric augmentation

The same validation set of **640 images** is used for comparison.

The comparison includes:

* PSNR
* SSIM
* LPIPS

The corresponding validation metric CSV files and generated graphs can be included in the project results if required for analysis and presentation.

---

## Hardware and Software

### Hardware

* NVIDIA GPU environment
* Google Colab used for model training and experimentation

### Software

* Python
* PyTorch
* NumPy
* scikit-image
* LPIPS
* Matplotlib
* Google Colab

---

## Offline Evaluation

The submitted solution is designed to satisfy the evaluation environment requirements.

The inference system:

* Does not require internet access
* Does not require API keys
* Does not download models during execution
* Does not require user interaction
* Does not require manual configuration
* Uses the model weights included in the repository
* Supports NVIDIA GPU execution
* Reads `.npy` files directly
* Generates `.npy` restored outputs

---

## Reproducibility

A fixed random seed of **42** is used during training to improve reproducibility.

The training configuration, model configuration and checkpoint information are stored with the trained model and training results.

---

## External Resource Disclosure

The project uses the PyTorch deep learning framework and the NAFNet architecture as the basis for image restoration.

The submitted repository contains the required model implementation and trained model weights for inference.

No external model download or API access is required during final evaluation.

---

## Team Members

**Team Name:** Wafer Endeavours

**Institution:** Chennai Institute of Technology

* M Pavithra
* R Pavithra 
* S Kanagalakshmi
* S Kanimozhi

---


The generated `test_output` directory will contain the restored `.npy` images with the same filenames as the corresponding input images.
