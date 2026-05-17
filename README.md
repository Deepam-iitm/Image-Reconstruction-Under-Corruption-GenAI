# 🧠 Generative AI for Image Reconstruction — NPPE

> **Kaggle Competition** · Apr 17–19, 2026 · Scored on Mean Squared Error (lower is better)

Reconstruct clean 32×32 RGB images from corrupted inputs using a two-stage NAFNet-based deep learning pipeline with EMA, mixed-precision training, and a combined perceptual loss objective.

---

## 🎯 Problem Statement

Each image in the dataset has been corrupted by a **random composition of multiple degradation types**. The corruption style and severity vary per image and are **not labeled** — the model must learn to undo all corruption patterns from training pairs alone.

Known degradation categories include:

| Category | Examples |
|---|---|
| **Masking** | Patch dropout, rectangular occlusions |
| **Noise** | Gaussian, salt-and-pepper, Poisson |
| **Blur** | Gaussian blur, motion blur |
| **Geometric** | Rotation, scaling, patch distortion |
| **Photometric** | Contrast shift, brightness change |

**Evaluation metric:** Mean Squared Error (MSE) between the reconstructed and hidden clean images. Range: [0, ∞), lower is better.

---

## 🗂️ Dataset

| File / Folder | Description |
|---|---|
| `train_corrupt/` | Corrupted training images (PNG, 32×32 RGB) |
| `train_clean/` | Corresponding clean target images |
| `train.csv` | Metadata — columns: `id`, `class`, `clean_filename`, `corrupt_filename` |
| `test_corrupt/` | Corrupted test images (PNG, 32×32 RGB) |
| `sample_submission.csv` | Submission template with correct format |

**Image spec:** 32×32 pixels · 3 channels (RGB) · Integer pixel values ∈ [0, 255] · 3072 values per image

**Submission format:** Flattened pixel values in CSV, one row per test image.

---

## 🔍 Approach

The solution uses a two-stage supervised training strategy on paired (corrupt → clean) images with no corruption labels required:

1. **Stage 1 — Perceptual Pretraining (80 epochs):**  Combined `L1 + MS-SSIM` loss to learn both pixel-level accuracy and structural similarity simultaneously. Cosine annealing with a brief linear warm-up stabilizes early training.

2. **Stage 2 — MSE Fine-tuning (10 epochs):**  Short fine-tune at a reduced learning rate using pure MSE loss to directly minimize the competition metric.

**Key training techniques:**
- Exponential Moving Average (EMA, decay = 0.999) of model weights for smoother inference
- Mixed-precision training (FP16 via `torch.amp`) for memory efficiency on T4 GPU
- Gradient clipping (max norm = 1.0) for stability
- Intermediate submissions at epochs 10 and 30 to track leaderboard vs. validation MSE correlation

---

## 🏗️ Model Architecture

The backbone is a **NAFNet (Nonlinear Activation Free Network)** — a lightweight yet powerful image restoration model originally proposed for denoising and deblurring.

```
Input (3 × 32 × 32)
        │
  ┌─────▼─────┐
  │  Stem Conv │  (3 → width)
  └─────┬─────┘
        │
  ┌─────▼──────────────────────────────────┐
  │  Encoder (4 stages, U-Net style)        │
  │   Stage 1: 2 NAF blocks  ──────────────┤→ skip₁
  │   Stage 2: 2 NAF blocks  ──────────────┤→ skip₂
  │   Stage 3: 4 NAF blocks  ──────────────┤→ skip₃
  │   Stage 4: 8 NAF blocks  ──────────────┤→ skip₄
  └─────┬───────────────────────────────────┘
        │
  ┌─────▼─────┐
  │  Middle    │  12 NAF blocks
  └─────┬─────┘
        │
  ┌─────▼──────────────────────────────────┐
  │  Decoder (4 stages, with skip fusions)  │
  │   Stage 1: 2 NAF blocks  ←─────────────┤ skip₄
  │   Stage 2: 2 NAF blocks  ←─────────────┤ skip₃
  │   Stage 3: 2 NAF blocks  ←─────────────┤ skip₂
  │   Stage 4: 2 NAF blocks  ←─────────────┤ skip₁
  └─────┬───────────────────────────────────┘
        │
  ┌─────▼─────┐
  │  Head Conv │  (width → 3)
  └─────┬─────┘
        │
  Output (3 × 32 × 32)   [residual added to input]
```

Each **NAF block** replaces all nonlinear activations with simple gating mechanisms (SimpleGate and Channel Attention), making the network fast and stable without BatchNorm.

**Model config:**

| Hyperparameter | Value |
|---|---|
| Base channel width | 48 |
| Encoder blocks | [2, 2, 4, 8] |
| Middle blocks | 12 |
| Decoder blocks | [2, 2, 2, 2] |

---

## 🔁 Training Pipeline

### Data

- Training images are auto-discovered from `train.csv`
- A held-out validation split of **2,000 samples** is used for MSE tracking
- Images are loaded as normalized float tensors in [0, 1]
- Standard augmentations applied during training (horizontal/vertical flips, random crops)

### Loss Functions

**Stage 1:**
```
Loss = (1 - ssim_weight) × L1(pred, target) + ssim_weight × (1 - MS_SSIM(pred, target))
```
with `ssim_weight = 0.2`.

**Stage 2:**
```
Loss = MSE(pred, target)
```

### Optimization

| Setting | Stage 1 | Stage 2 |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 2e-4 | 1e-5 |
| Weight decay | 1e-4 | — |
| LR schedule | Cosine + warm-up (3%) | Constant |
| Epochs | 80 | 10 |
| Batch size | 128 | 128 |

---

## ⚙️ Experiment Configuration

All hyperparameters are stored in `config.json` at the start of each run for full reproducibility:

```json
{
  "val_size": 2000,
  "batch_size": 128,
  "num_workers": 4,
  "width": 48,
  "enc_blks": [2, 2, 4, 8],
  "middle_blk": 12,
  "dec_blks": [2, 2, 2, 2],
  "stage1_epochs": 80,
  "stage1_lr": 0.0002,
  "stage1_wd": 0.0001,
  "stage1_warmup_frac": 0.03,
  "ssim_weight": 0.2,
  "stage2_epochs": 10,
  "stage2_lr": 1e-05,
  "ema_decay": 0.999,
  "grad_clip": 1.0,
  "intermediate_submit_epochs": [10, 30]
}
```

## 📦 Requirements

| Package | Purpose |
|---|---|
| `torch` ≥ 2.0 | Model training and inference |
| `torchvision` | Image transforms |
| `numpy` | Array operations |
| `pandas` | CSV I/O |
| `Pillow` | Image loading |
| `matplotlib` | Visualization |

**Hardware:** Trained on a single NVIDIA Tesla T4 (15.6 GB VRAM) via Kaggle Notebooks with GPU acceleration enabled.

