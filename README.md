# March Machine Learning Mania 2026 Winning Solution

This repository is a cleaned, GitHub-ready package built from the archived project files. Unnecessary notebooks, logs, helper scripts, and raw data dumps were removed.

## What is included

- `generate_submission.py` - cleaned training/inference pipeline reconstructed from the archived code
- `reproduce_reference_submission.py` - writes the exact archived winning `submission.csv`
- `verify_submission.py` - compares any generated submission against the archived reference
- `submission_reference.csv` - exact archived final submission bundled for verification
- `CHECKSUMS.txt` - MD5 checksum for the archived final submission
- `requirements.txt` - pinned package versions used for the cleaned rerun

## Important note on exact reproducibility

The bundled `submission_reference.csv` is the exact archived final output recovered from the source zip.

The cleaned training script (`generate_submission.py`) reproduces the feature engineering and model pipeline, but gradient-boosting outputs are version-sensitive. In this environment, rerunning the pipeline produced a close result but not a byte-for-byte match to the archived submission. To guarantee an exact match for GitHub/release purposes, use `reproduce_reference_submission.py`.

## Data setup for training reruns

Download the Kaggle competition data and place the CSV files under:

```text
./data/march-machine-learning-mania-2026/
```

Expected folder layout:

```text
project/
├── data/
│   └── march-machine-learning-mania-2026/
│       ├── MRegularSeasonDetailedResults.csv
│       ├── MNCAATourneyDetailedResults.csv
│       ├── MNCAATourneySeeds.csv
│       ├── MTeamConferences.csv
│       ├── MMasseyOrdinals.csv
│       ├── WRegularSeasonDetailedResults.csv
│       ├── WNCAATourneyDetailedResults.csv
│       ├── WNCAATourneySeeds.csv
│       ├── WTeamConferences.csv
│       └── SampleSubmissionStage2.csv
├── CHECKSUMS.txt
├── generate_submission.py
├── reproduce_reference_submission.py
├── submission_reference.csv
├── verify_submission.py
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Option 1: write the exact archived submission

```bash
python reproduce_reference_submission.py --output submission.csv
```

Then verify the checksum or compare against the bundled reference:

```bash
python verify_submission.py --generated submission.csv --reference submission_reference.csv
```

## Option 2: rerun the cleaned model pipeline

```bash
python generate_submission.py --data-dir ./data/march-machine-learning-mania-2026 --output submission.csv
```

## Files intentionally removed

The following were removed from the GitHub package because they are not required for reproducibility or release:

- exploratory notebook copies
- local execution logs and error captures
- one-off helper scripts used only during archive cleanup
- the raw Kaggle competition dataset

## Verification

Archived reference MD5:

```text
d9775d697dd780394e93cba765be196a
```
