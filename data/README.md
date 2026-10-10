# Data directory

Only two **synthetic** files are shipped (`sample.csv`, `sample_acoustic.csv`); regenerate them with `python -m mindtext make-sample --task text|tabular --out ...`.
Real datasets are **not** included (size and/or licence). Everything else in this folder is git-ignored.

## Reddit text → `data/reddit_mental_health.csv`
1. Obtain the Reddit mental-health dataset from the source you used on Kaggle (`entenam/reddit-mental-health-dataset` in your notebooks) and check its licence/citation requirements on that page.
2. `python -m mindtext prepare-reddit --raw-dir "<path>/Original Reddit Data/raw data" --out data/reddit_mental_health.csv`
   (prints files that match no label rule instead of silently dropping them; writes `.ingest.json` with per-file counts).
3. `python -m mindtext data-check --data data/reddit_mental_health.csv --config configs/default.yaml`
4. `python -m mindtext train --config configs/default.yaml --data data/reddit_mental_health.csv`

## E-DAIC acoustic features → `data/edaic_features.csv`
1. Obtain E-DAIC from its official distribution (`https://dcapswoz.ict.usc.edu/wwwedaic/` is what your notebooks used) and accept its data-use terms.
2. Lay out `raw_egemaps/<id>_P/<id>_OpenSMILE2.3.0_egemaps.csv` and `labels/{train,dev,test}_split.csv`.
3. `python -m mindtext build-acoustic-features --raw-dir <raw_egemaps> --labels-dir <labels> --out data/edaic_features.csv` (add `--sources egemaps mfcc` for a 186-style set; `--sentinel --nan-fill 0` mimics the first notebooks' handling of -201 and all-missing statistics).
4. `python -m mindtext train --config configs/acoustic.yaml --data data/edaic_features.csv`
