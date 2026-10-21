# Reinke's edema classification: R1 code and aggregate results

This repository contains the analysis code, recorded run settings, and aggregate results for the R1 revision. It contains no voice recordings, feature tables, split manifest, participant-level predictions, or participant identifiers. It has an independent Git history that does not include the private project's clinical data.

## What can be reproduced publicly

The saved aggregate results can be inspected directly in `results_r1_nested/`, `results_r1_nested/optimizer_search/`, `results_r1_fixed_relu/`, and `results_r1_fixed_relu_multiseed/`. Run `python3 scripts/verify_public_package.py` to check the distributed files and their SHA-256 hashes. The classifier heatmap can be regenerated from the aggregate metrics with `python3 scripts/generate_r1_figures.py` after installing `requirements-r1.txt`.

The code documents the full evaluation and optimization procedure. Exact retraining, participant-level metric recalculation, and regeneration of the calibration and confusion-matrix figures require restricted clinical data or derived feature tables. They cannot be performed from this public package alone. Access to the source recordings and derived clinical tables is governed by the original ethics and data-use constraints.

## Run with separately authorized inputs

Use Python 3.10 and install the analysis dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-r1.txt
python scripts/verify_public_package.py
bash run_r1.sh /secure/path/to/features_win
```

The private `features_win` directory must contain a window manifest named `manifest.csv` and the five representation tables `fft_topk_hann.csv`, `mfcc_stats.csv`, `spectral_stats.csv`, `wav2vec2.csv`, and `wpd_haar7_std.csv`. The 768 embedding columns in the legacy `wav2vec2.csv` filename come from frozen `microsoft/wavlm-base`, not wav2vec 2.0. The immutable checkpoint revision used in the original extraction was not recorded, so feature extraction is not claimed to be bitwise reproducible. `prepare_dataset.py` and `requirements-extraction.txt` document feature extraction for researchers who have authorized audio access.

`run_r1.sh` writes outputs under the ignored `reproduced_r1/` directory. The nested ReLU model family uses initialization seeds 17, 37, 59, 83, and 109. The separate fixed ReLU sensitivity analysis uses 42, 123, 2026, 3407, and 8675309. The fixed logistic reference uses seed 42. The optimizer search uses 200 objective calls per outer split and is computationally expensive. The supplied requirements reflect the local R1 environment and have not been checked on a fresh machine.

The analysis covers 36 subjects (20 controls and 16 cases) over 851 windows, with three repeats of five subject-grouped outer folds and four inner folds. Repeated folds and seeds describe variation on this one cohort; they are not independent patients or external validation.

## Published results

- `results_r1_nested/aggregate_metrics.csv`: classifier and representation summaries across three outer repeats.
- `results_r1_nested/optimizer_search/optimizer_aggregate_metrics.csv`: six optimizer summaries.
- `results_r1_fixed_relu/aggregate_metrics.csv`: fixed ReLU reference summaries.
- `results_r1_fixed_relu_multiseed/aggregate_metrics.csv` and `seed_aggregate_metrics.csv`: fixed ReLU seed sensitivity summaries.
- Each results directory includes its run configuration and, where recorded, run metadata. These are descriptive records, not clinical input data.

No participant-level files are present in this public history. The source data and detailed private outputs remain outside this repository.
