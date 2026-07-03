# Reproducibility

## 1. Prepare official data

Place the official LRA archive outside this repository and run:

```bash
python scripts/import_official_lra_download.py --help
python scripts/prepare_lra_official.py --help
```

Do not use synthetic data for paper results. Compare the resulting fingerprints
with `paper_artifacts/data_quality_report.json`.

## 2. Run a paper cell

```bash
python scripts/train_lra_light.py \
  --task listops --variant dtr --seed 0 \
  --config configs/lra_listops_4090_tuned_journal.yaml \
  --model-config configs/model_b2s6_1660ti.yaml \
  --method-config configs/method_variants.yaml \
  --data-source official_lra --device cuda --require-cuda \
  --amp --amp-dtype bf16 --deterministic
```

Use the corresponding task configuration for Text and Pathfinder and seeds 0--4.

## 3. Regenerate analysis

Download the Zenodo evidence archive, verify `SHA256SUMS.txt`, and place the
snapshot at `evidence_snapshot/`. Then run the scripts in `analysis/` in this
order: `extract_results.py`, `render_tables.py`, and `render_figures.py`.

## Environment limitation

The artifacts preserve Python, PyTorch, CUDA, GPU, precision, and OS metadata.
The server backup did not preserve a complete `pip freeze`; therefore exact
versions of unrecorded transitive packages cannot be claimed. This limitation is
stated rather than filled with reconstructed guesses.
