# VolDiT: Controllable Volumetric Medical Image Synthesis with Diffusion Transformers

This repository contains the official implementation of **VolDiT**, a latent diffusion framework for 3D medical image synthesis based on Diffusion Transformers.
The pipeline consists of a **VQ-GAN** autoencoder (Stage 1) that compresses 3D volumes into a compact latent space,
followed by a **VolDiT** diffusion model (Stage 2) that operates in that latent space.
Conditional generation is supported via **TGCA (Timestep-Gated Control Adapter)**, which extends the frozen VolDiT base model with mask-guided control without modifying its weights.
![VolDiT sample sweep](assets/sample_1_0_spotlight.gif)
> **VolDiT: Controllable Volumetric Medical Image Synthesis with Diffusion Transformers**
> arXiv: [2603.25181](https://arxiv.org/abs/2603.25181)

---

## Abstract

Diffusion models have become a leading approach for high-fidelity medical image synthesis. However, most existing methods for 3D medical image generation rely on convolutional U-Net backbones within latent diffusion frameworks. While effective, these architectures impose strong locality biases and limited receptive fields, which may constrain scalability, global context integration, and flexible conditioning. In this work, we introduce VolDiT, the first purely transformer-based 3D Diffusion Transformer for volumetric medical image synthesis. Our approach extends diffusion transformers to native 3D data through volumetric patch embeddings and global self-attention operating directly over 3D tokens. To enable structured control, we propose a timestep-gated control adapter that maps segmentation masks into learnable control tokens that modulate transformer layers during denoising. This token-level conditioning mechanism allows precise spatial guidance while preserving the modeling advantages of transformer architectures. We evaluate our model on high-resolution 3D medical image synthesis tasks and compare it to state-of-the-art 3D latent diffusion models based on U-Nets. Results demonstrate improved global coherence, superior generative fidelity, and enhanced controllability. Our findings suggest that fully transformerbased diffusion models provide a flexible foundation for volumetric medical image synthesis.

![VolDiT abstract overview](assets/abstract.jpg)




---

## Architecture

- **Stage 1 — VQ-GAN**: Compresses 3D CT volumes (e.g. 512×512×256) into 8-channel latent codes at 8× spatial downsampling using an EMA codebook. A 512×512×256 input becomes a [8, 64, 64, 32] latent tensor.
- **Stage 2 — VolDiT**: A 3D Diffusion Transformer that tokenises the latent volume into non-overlapping p×p×p patches. Each token is processed through L DiT blocks with adaptive layer normalisation (AdaLN) conditioned on the diffusion timestep. Fixed 3D sinusoidal positional encodings are used. Training uses a cosine noise schedule with v-prediction and Smooth L1 loss, T=300 timesteps.
- **TGCA (Timestep-Gated Control Adapter)**: Wraps the frozen VolDiT base model for conditional generation. A lightweight adapter branch processes the condition (e.g. segmentation mask) and injects control signals into the frozen DiT blocks via timestep-dependent gating: γ(t) = σ(MLP(t)). The final projection is zero-initialised for stable training from the pretrained VolDiT weights. TGCA runs the full denoising pass internally — no separate frozen model is needed at inference.

### Model Configs

The transformer configs in `configs/transformer/` use the `dit.params` and `dit.scheduler` sections. They are named by VQ-GAN downsampling factor and patch size:

| Config | Model | Patch size | Layers | Hidden dim | Heads |
|--------|-------|------------|--------|------------|-------|
| `dit_ds8_xs2.yaml` | VolDiT-XS | 2 | 6 | 384 | 6 |
| `dit_ds8_xs4.yaml` | VolDiT-XS | 4 | 6 | 384 | 6 |
| `dit_ds8_s2.yaml` | VolDiT-S | 2 | 12 | 384 | 6 |
| `dit_ds8_s4.yaml` | VolDiT-S | 4 | 12 | 384 | 6 |
| `dit_ds8_b2.yaml` | VolDiT-B | 2 | 12 | 768 | 12 |
| `dit_ds8_b4.yaml` | VolDiT-B | 4 | 12 | 768 | 12 |
| `dit_ds8_l2.yaml` | VolDiT-L | 2 | 24 | 1152 | 16 |
| `dit_ds8_l4.yaml` | VolDiT-L | 4 | 24 | 1152 | 16 |

Patch sizes p=2 and p=4 are supported. Larger patch sizes reduce the number of tokens and accelerate training on high-resolution volumes.

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- MONAI
- xformers (optional, for flash attention)
- nibabel, omegaconf, timm, pandas

Install via conda:

```bash
conda env create -f environment.yml
conda activate dit_gen
```

---

## Pretrained Checkpoints

Pretrained VQ-GAN autoencoder and unconditional VolDiT weights for the public LUNA16 setup are available on Hugging Face:

```bash
hf download AICM-HD/voldit --local-dir checkpoints/
```

Use the downloaded autoencoder checkpoint as `--stage1_ckpt` / `--vqvae_ckpt` and the unconditional VolDiT checkpoint as `--diff_ckpt` / `--dit_ckpt` in the sampling and TGCA commands below.

---

## Data Preparation

Training expects CSV files with a column named `image` containing absolute paths to NIfTI (`.nii` / `.nii.gz`) CT volumes.
Images are clipped to HU `[-1000, 1000]` and scaled to `[-1, 1]`.

```
image
/data/ct_001.nii.gz
/data/ct_002.nii.gz
...
```

Split into a training CSV and a validation CSV before starting.

The public release targets **LUNA16** lung CT volumes at 512×512×256 resolution.

---

## Training: Unconditional VolDiT

Training proceeds in two stages: first the VQ-GAN autoencoder, then VolDiT in the compressed latent space.

### Stage 1 — Train VQ-GAN

```bash
torchrun --nproc_per_node=2 src/scripts/train_vqgan.py \
    --config configs/stage1/vqgan_ds8.yaml \
    --training_ids ids/train.csv \
    --validation_ids ids/val.csv \
    --output_dir outputs/ \
    --run_name vqgan_v1
```

The best checkpoint is saved to `outputs/vqgan_v1/best_model.pth`.

---

### Stage 2a — Pre-encode Images to Latents (recommended)

Pre-encoding avoids redundant VQ-GAN forward passes during VolDiT training.

```bash
python src/scripts/encode_images.py \
    --csv ids/train.csv \
    --output_dir data/latents/train/ \
    --vqvae_ckpt outputs/vqgan_v1/best_model.pth \
    --config configs/stage1/vqgan_ds8.yaml \
    --batch_size 1 \
    --device cuda
```

Repeat for the validation set. Each run produces a `latents.csv` in the output directory.
Pass these CSVs as `--training_ids` and `--validation_ids` when training VolDiT.

Already-encoded files are automatically skipped on re-runs.

Compute the latent scale factor (used to normalise the latent distribution to unit variance):

```bash
python src/scripts/compute_scale_factor.py \
    --latents_csv data/latents/train/latents.csv \
    --limit 200
```

Set the printed `scale_factor` in your DiT config under `training.scale_factor`.


---

### Stage 2b — Train VolDiT

Trains a VolDiT model in the VQ-GAN latent space using a cosine noise schedule with v-prediction.

```bash
torchrun --nproc_per_node=2 src/scripts/train_dit.py \
    --config configs/transformer/dit_ds8_l4.yaml \
    --training_ids data/latents/train/latents.csv \
    --validation_ids data/latents/val/latents.csv \
    --output_dir outputs/ \
    --run_name dit_v1
```

To train without precomputed latents (online VQ-GAN encoding during training):

```bash
torchrun --nproc_per_node=2 src/scripts/train_dit.py \
    --config configs/transformer/dit_ds8_l4.yaml \
    --no_precomputed_latents \
    --vqvae_ckpt outputs/vqgan_v1/best_model.pth \
    --config_vqvae configs/stage1/vqgan_ds8.yaml \
    --training_ids ids/train.csv \
    --validation_ids ids/val.csv \
    --output_dir outputs/ \
    --run_name dit_v1
```

The best EMA checkpoint is saved to `outputs/dit_v1/best_model.pth`.

The scripts provide default `training` and `optim` values if those sections are omitted from the transformer config. Add those sections to the YAML when you want run-specific overrides.

---

### Unconditional Sampling

```bash
python src/scripts/sample_dit.py \
    --stage1_ckpt outputs/vqgan_v1/best_model.pth \
    --stage1_cfg configs/stage1/vqgan_ds8.yaml \
    --diff_ckpt outputs/dit_v1/best_model.pth \
    --diff_cfg configs/transformer/dit_ds8_l4.yaml \
    --latent_shape 64 64 32 \
    --output_dir samples/ \
    --n_samples 4 \
    --timesteps 300 \
    --scheduler ddpm \
    --scale_factor 7.87
```

`--latent_shape` must match the spatial dimensions of the encoded latents (input volume spatial size divided by 8 for the ds8 VQ-GAN; e.g. 512×512×256 → 64×64×32).

Outputs are saved as `.nii.gz` files with HU values. EMA weights are used automatically if available in the checkpoint.

To sample across multiple training checkpoints (epoch-range mode):

```bash
python src/scripts/sample_dit.py \
    --stage1_ckpt outputs/vqgan_v1/best_model.pth \
    --stage1_cfg configs/stage1/vqgan_ds8.yaml \
    --diff_run_dir outputs/dit_v1/ \
    --diff_cfg configs/transformer/dit_ds8_l4.yaml \
    --epoch_start 100 \
    --epoch_end 500 \
    --epoch_step 100 \
    --latent_shape 64 64 32 \
    --output_dir samples/
```

---

## Training: Conditional VolDiT (TGCA)

TGCA extends the trained VolDiT with mask-guided generation. The VolDiT weights are frozen; only the TGCA adapter is trained.

**Design note:** TGCA wraps the frozen VolDiT base model and runs the full denoising forward pass internally — no separate frozen diffusion model is needed at training or inference time.

### Stage 3a — Pre-encode Images + Masks

Prepare a CSV with one column per condition key in addition to `image`:

```
image,mask
/data/ct_001.nii.gz,/masks/ct_001_seg.nii.gz
...
```

Then encode:

```bash
python src/scripts/encode_images_cond.py \
    --csv ids/train_cond.csv \
    --output_dir data/latents_cond/train/ \
    --vqvae_ckpt outputs/vqgan_v1/best_model.pth \
    --config configs/stage1/vqgan_ds8.yaml \
    --condition_keys mask \
    --device cuda
```

Produces `tgca_latents.csv` with paths to the encoded image latent and all preprocessed mask tensors.
Run for both train and validation sets.

---

### Stage 3b — Train TGCA

```bash
torchrun --nproc_per_node=2 src/scripts/train_tgca.py \
    --config configs/transformer/dit_ds8_l4.yaml \
    --tgca_config configs/tgca/tgca_ds8.yaml \
    --dit_ckpt outputs/dit_v1/best_model.pth \
    --training_ids data/latents_cond/train/tgca_latents.csv \
    --validation_ids data/latents_cond/val/tgca_latents.csv \
    --output_dir outputs/ \
    --run_name tgca_v1
```

`--config` must be the same DiT architecture config used to train the checkpoint passed via `--dit_ckpt`. TGCA does not duplicate pretrained DiT model parameters in its own config.

Key config parameters in `configs/tgca/tgca_ds8.yaml`:
- `tgca.condition_keys` — list of mask column names matching the encode step
- `tgca.params.condition_channels` — total number of channels after stacking the condition keys
- `tgca.params.inject_layers` — `null` to inject into all DiT blocks, or an integer `N` to inject only into the last `N` blocks
- `tgca.params.finetune_last_n_blocks` — unfreeze the last N DiT blocks in addition to the adapter (0 = freeze all)
- `training.condition_dropout` — probability of dropping the entire condition during training

---

### Conditional Sampling

Provide a CSV with one row per subject and one column per condition key pointing to the original NIfTI mask files.
Note: pass the original NIfTI CSV (not the precomputed latents CSV), since masks are loaded and preprocessed at inference time.

```bash
python src/scripts/sample_tgca.py \
    --stage1_ckpt outputs/vqgan_v1/best_model.pth \
    --stage1_cfg configs/stage1/vqgan_ds8.yaml \
    --diff_cfg configs/transformer/dit_ds8_l4.yaml \
    --dit_ckpt outputs/dit_v1/best_model.pth \
    --tgca_ckpt outputs/tgca_v1/best_model.pth \
    --tgca_cfg configs/tgca/tgca_ds8.yaml \
    --csv ids/test_cond.csv \
    --condition_keys mask \
    --latent_shape 64 64 32 \
    --roi_size 512 512 256 \
    --output_dir samples/cond/
```

`--roi_size` must match the spatial size used during training so that masks are resized consistently.
One `.nii.gz` volume is generated per CSV row. EMA weights are loaded automatically if present in the checkpoint.

---

## Distributed Training

All training scripts support multi-GPU training via PyTorch DDP. Use `torchrun`:

```bash
torchrun --nproc_per_node=<N_GPUS> src/scripts/train_vqgan.py \
    --config configs/stage1/vqgan_ds8.yaml \
    --training_ids ids/train.csv \
    --validation_ids ids/val.csv \
    --output_dir outputs/ \
    --run_name vqgan_v1
```

The same applies to `train_dit.py` and `train_tgca.py`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{voldit2026,
  title     = {VolDiT: Controllable Volumetric Medical Image Synthesis with Diffusion Transformers},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
}
```

---

## License

This code is licensed under the Apache License 2.0. See the LICENSE file for details.
