# Delta Trust-Region Control for Selective SSMs

This repository contains the exact training implementation used for the paper
*Length-Aware Trust-Region Control of Discretization Drift in Selective State Space Models*.

## Scope

The paper evaluates `base`, `fixed_delta`, `dtr`, and `dtrl` on official LRA
ListOps, Text, and Pathfinder data with five seeds. The reported setup is a
compute-controlled lightweight evaluation, not a reproduction of the canonical
LRA leaderboard configuration.

## Install

Use Python 3.12.3 and a CUDA 12.8 build of PyTorch 2.10.0. Install the remaining
dependencies with `pip install -r requirements.txt`. See
`environment/verified_runtime.json` for the environment fields preserved by the
run artifacts.

## Data

Official LRA files are not redistributed. Follow `REPRODUCIBILITY.md` to import
and preprocess the official release. Dataset fingerprints in `paper_artifacts/`
allow prepared splits to be checked against the paper evidence.

## Paper configurations

- `configs/lra_listops_4090_tuned_journal.yaml`
- `configs/lra_text_4090_tuned_journal.yaml`
- `configs/lra_pathfinder_4090_tuned_journal.yaml`
- `configs/model_b2s6_1660ti.yaml`
- `configs/method_variants.yaml`

## Evidence

The complete evidence and checkpoint archives are distributed separately through
Zenodo because they are not appropriate for Git history. `paper_artifacts/`
contains the manifest, analysis result, and data/protocol audit reports.

## License

Code is released under the MIT License. Dataset use remains governed by the
original LRA sources.
