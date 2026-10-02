# Evaluation and analysis tools

Install core metric/plotting dependencies with:

```bash
python -m pip install -e '.[evaluation]'
```

This includes SciPy, matplotlib, scikit-learn, PRDC and OpenCV. No evaluation
package is needed to train/sample the DiT beyond the core dependencies.

Optional external systems:

- Whole/segmented FRD: `python -m pip install frd-score`. The imported wrapper
  identifies its reference as https://github.com/RichardObi/frd-score. FRD has
  API variants and requires PyRadiomics; verify the installed API and protocol.
  This consolidation does not vendor a private FRD checkout or claim backend
  compatibility was tested. An explicit error reports a missing backend.
- Organ segmentation: `python -m pip install TotalSegmentator`, then follow
  https://github.com/wasserth/TotalSegmentator for model weights and usage.
  Or supply precomputed masks through `--seg_dir`. Weights and segmentations
  belong outside the checkout.
- Feature extractors use Torch Hub sources `Warvito/MedicalNet-models` and
  `Warvito/radimagenet-models`; first use downloads external weights. Set
  `TORCH_HOME` to an external cache. Record a revision and weight hashes when
  using them for a reproducible experiment.

All metric scripts require `--output_dir`. CSV inputs, generated TSV/NPZ,
segmentation masks, figures and videos stay under an external run directory.
Examples (with DATA_ROOT/RUN_ROOT from the root README):

```bash
python evaluation/compute_hu_distribution.py \
    --real_csv "$DATA_ROOT/ids/test.csv" \
    --fake_dir "$RUN_ROOT/samples" \
    --output_dir "$RUN_ROOT/evaluation/hu"
python evaluation/plot_metrics.py --metrics-root "$RUN_ROOT/evaluation"
python evaluation/model_selection.py --metrics-root "$RUN_ROOT/evaluation"
python evaluation/make_sweep_videos.py \
    --input-dir "$RUN_ROOT/samples" --output-dir "$RUN_ROOT/videos"
```

Plot/model-selection scripts expect the existing per-model/per-epoch metric tree;
run `--help` and inspect the metric field names. The model-selection composite is
an exploratory relative score, not a clinical or statistically validated ranking.

Known protocol issues retained for further research: 2.5D feature extraction
concatenates slices while filename lists represent volumes; PRDC stacking assumes
aligned feature counts. Do not interpret slices as independent patients or use
that path without validating sample alignment. Evaluation resampling differs from
some training/encoding paths. ImageNet/RadImageNet preprocessing assumptions and
feature weights need verification. FRD defaults include abdominal organs and may
need adaptation for chest CT. Segmentation-derived volumes depend on correct
physical output geometry. Report failed evaluations and keep per-volume IDs.
