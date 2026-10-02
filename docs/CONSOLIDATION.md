# Consolidation provenance and supported features

This source consolidation starts from `Cardio-AI/voldit` commit
`76c7063` (public repository main fetched 2026-10-02) and selectively integrates
`VolDiT_v2` commit `c54ee9ff70960041824c8aa9f222837ef4e667f9`, the older home
source copy, and `voldit_improved` analysis tools. The original checkouts and
complete preservation inventory remain external to this repository.

| Source | Destination / disposition |
| --- | --- |
| Public VolDiT | Retained VQGAN, encoding, canonical TGCA, config helpers, published configs and license; retained lazy optional VQGAN loss fixes. |
| VolDiT_v2 DiT, EMA, trainer, train/sample entrypoints | Merged into their corresponding `src/` files. Includes diffusion/FM, self-conditioning, configurable dropout, spatial spacing, EMA warmup, Min-SNR, offset noise, accumulation, normalization and LR/weight decay options. |
| VolDiT_v2 schedulers | Added `flow_matching_scheduler.py` and `dpm_solver_scheduler.py`. |
| VolDiT_v2 dataloading | Added `AugmentedLatentDataset` and its config option while retaining TGCA interfaces. |
| VolDiT_v2 `configs/stage1/vqgan_ds8.yaml` | Preserved as `configs/stage1/vqgan_ds8_development.yaml`; codebook size differs from the public config. Other stage1 and transformer configs retain their names. |
| VolDiT_v2 ControlNet implementation | Superseded by the public TGCA implementation, which already contains learned timestep gates, adapter scales and token-grid resizing missing from the development implementation. No second competing adapter code path. |
| VolDiT_v2 ControlNet config pair | Converted to `configs/tgca/tgca_development{,_test}.yaml`. Pass the same file as `--config` and `--tgca_config`; sampler uses it as both `--diff_cfg` and `--tgca_cfg`. Parameter names are translated, scientific values retained. |
| Older home baseline / improved configuration | Exact variants in `configs/experiments/historical_home/`; source-only historical settings. `01_improved.yaml` has conflicting model/training self-conditioning flags and must be corrected explicitly before use. |
| VolDiT_v2 evaluation | Added the full `evaluation/` source tree. Metrics and downloaded evaluation models remain external. |
| `voldit_improved` | Added `evaluation/{plot_metrics,model_selection,make_sweep_videos}.py`; all input/output locations are caller selected. |
| CardioDiT local `src/models/scheduler.py` | Existing zero-terminal-SNR rescaling block copied into the shared VolDiT schedule implementation, because VolDiT development configs referenced an option their scheduler did not accept. Defaults preserve the original schedule. |
| Ignored development notes / IDE files / launch settings | Preserved externally; not runtime dependencies. Their scientific improvement claims are not adopted as validated results. |

## Integration corrections

- Corrected PyTorch attention layouts to attend over tokens, with explicit
  reference parity for both SDPA and manual implementations. xformers is optional.
- Sampling now propagates self-conditioning and resets multistep solver history
  for every sample batch. DPM++ explicitly returns the final clean prediction.
- Rejected unsupported subsampled DDPM: its posterior implements adjacent training
  steps. DDIM / DPM++ remain available for fewer function evaluations.
- Retained support for public and development configuration schemas. Objective
  and LR scheduler names fail clearly when unknown.
- Checkpoints retain normalization, optimizer, scaler, LR and EMA state, including
  the final checkpoint. Resume restores normalization and scaler state.
- Corrected partial gradient-accumulation windows and skipped-step EMA accounting.
  Imported unconditional trainer validation and latent moments reduce across ranks.
- CPU checks disable CUDA AMP; public GPU training behavior remains available.
- Tensor/NumPy latent files load on CPU without dependence on NumPy private aliases.
- Generated media, CSV/TSV manifests, model artifacts and local developer checks
  are excluded from the source tree. Existing research configs remain source.

## Validation performed

Temporary checks run outside the repository on CPU and one RTX 4090 passed:
50 Python files parsed; 50 YAML configs loaded with constructor/signature checks;
attention compared with an explicit multi-head reference; diffusion/FM clean
estimate and Euler endpoint identities; frozen tiny VQGAN encode/decode with
unchanged weights/buffers; TGCA forward with token-grid resizing; and four tiny
training cases (DDPM/FM, self-conditioning off/on) with real parameter updates,
checkpoint/resume and decoded sampling. DDPM, DDIM, DPM++ and flow sampling were
covered, including multiple batches with a shorter final batch. Checks used
PyTorch 2.10.0, MONAI 1.5.2, OmegaConf 2.3.0, timm 1.0.22, NumPy 2.x,
pandas 2.3.3 and nibabel 5.3.3 in an existing environment. All 21 model/evaluation
command entrypoints also passed `--help` import/argument checks. They used synthetic
inputs and randomly initialized tiny models, not a pretrained quality benchmark.

## Limits of consolidation

The integrated options have not been shown to improve generation quality.
Small synthetic numerical checks establish execution and selected invariants,
not convergence, historical checkpoint provenance, clinical accuracy or dataset
geometry. Production-size distributed training and metric backends require
separate validation. Existing preprocessing variants (resize vs crop; per-image
intensity normalization vs fixed HU inversion) are preserved, not reconciled.
Use matched encoder/config/preprocessing and verify output affine/spacing for
physical measurements.

Latent flips are experimental: learned convolutions are not generally flip
equivariant. Historical configs and checkpoint inference settings must be paired
explicitly. Reduced-step DDIM/DPM++ grids retain historical discretization.
The TGCA path is diffusion-only and does not support self-conditioned bases or
normalized-latent training; the independent unconditional features are retained.
Use the unconditional runner for those options. The old TGCA/VQGAN distributed
trainer paths are retained and were not comprehensively revalidated here.

Evaluation limitations are listed in `EVALUATION.md`. Research claims and known
limitations are not converted into default scientific choices by this merge.
