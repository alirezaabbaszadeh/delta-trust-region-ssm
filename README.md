# Measuring and Constraining Optimizer-Induced Discretization Drift in Selective State Space Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Release](https://img.shields.io/badge/release-v1.0.0-green.svg)](https://github.com/alirezaabbaszadeh/delta-trust-region-ssm/releases/tag/v1.0.0)

This repository contains the verified code, configurations, and reproducibility tools for the paper:
> **"Measuring and Constraining Optimizer-Induced Discretization Drift in Selective State Space Models"**  
> *Alireza Abbaszadeh, Mohammad Hossein Moattar*  
> *Department of Computer Engineering, Mashhad Branch, Islamic Azad University, Mashhad, Iran*

---

## Overview

The study examines optimization dynamics in selective state space models (SSMs) where standard parameter updates can induce batch-dependent discretization drift. We evaluate:
- `base`: Standard selective SSM (Mamba-style) baseline.
- `fixed_delta`: Baseline with frozen discretization parameter group.
- `dtr`: Delta Trust Region with sampled realized-drift line-search control.
- `dtrl`: Length-aware Delta Trust Region adapting radius to sequence length.

All experiments are conducted across official Long Range Arena (LRA) benchmarks (**ListOps**, **Text**, and **Pathfinder**) over five matched random seeds (0–4) in a compute-controlled, lightweight protocol.

---

## Repository Structure

- `src/`: Core implementation of selective SSM layers and DTR trust-region controllers.
- `configs/`: Experiment configurations for ListOps, Text, and Pathfinder.
- `scripts/`: Data ingestion, training, and verification entry points.
- `analysis/`: Statistical extraction, Holm-Bonferroni correction, and visualization routines.
- `paper_artifacts/`: Data quality reports, protocol parity audits, and run manifests.

---

## Quickstart & Reproducibility

### 1. Environment Setup
Requires Python 3.12.3 and PyTorch 2.10.0+cu128:
```bash
pip install -r requirements.txt
```

### 2. Official LRA Data
Follow `REPRODUCIBILITY.md` to import and preprocess the official LRA datasets.

### 3. Training
Run a paper cell (e.g., ListOps with DTR on Seed 0):
```bash
python scripts/train_lra_light.py \
  --task listops --variant dtr --seed 0 \
  --config configs/lra_listops_4090_tuned_journal.yaml \
  --model-config configs/model_b2s6_1660ti.yaml \
  --method-config configs/method_variants.yaml \
  --data-source official_lra --device cuda --require-cuda \
  --amp --amp-dtype bf16 --deterministic
```

---

## Zenodo Evidence Package

The frozen execution artifacts, checkpoints, dataset fingerprints, and statistical outputs are distributed separately through Zenodo because they are not appropriate for Git history:
- Complete package ready at `release/zenodo_v1/`.
- Checksums and dataset hashes match `paper_artifacts/SHA256SUMS.txt`.

---

## Citation

```bibtex
@article{abbaszadeh2026measuring,
  title={Measuring and Constraining Optimizer-Induced Discretization Drift in Selective State Space Models},
  author={Abbaszadeh, Alireza and Moattar, Mohammad Hossein},
  journal={Information Sciences},
  year={2026}
}
```

## License

Code is released under the [MIT License](LICENSE). Dataset use remains governed by the original LRA sources.
