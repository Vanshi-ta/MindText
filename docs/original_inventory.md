# Inventory of the supplied materials

Ten notebooks were supplied (copied unmodified to `notebooks/original/`, original file names kept, including the `__1_`/`__2_` suffixes added by the download).
**No data files, saved models, README files, or plots outside the notebooks were supplied.** The notebooks use two *different* datasets for two *different* tasks; they were never merged.

| notebook | track | what it does | status |
|---|---|---|---|
| `preprocessing-reddit-data__2_` | Reddit | walks raw monthly CSVs, labels by file-name prefix, cleans text, writes `cleaned_labelled_raw_data.csv`; second cell drops `unknown` | ran; label-mapping flaw (see `original_results.md` C-1) |
| `reddit-dataset-model__2_` | Reddit | repeats the prep cells, EDA (class counts, length box-plot, word cloud), 80/20 split, TF-IDF(5000), LR / NB / LinearSVC / RF / LightGBM / MLP, bar chart | MLP cell never ran; LightGBM CM cell errored then was re-run; chart uses typed-in numbers; reads a user-uploaded processed CSV |
| `EDAIC_EDA__1_` | E-DAIC | downloads label CSVs + one participant archive (300_P), checks split disjointness, label/PHQ/PTSD/gender distributions, frame structure of eGeMAPS & MFCC files | exploratory only; last cells use variables (`duration`, `y`, `sr`) defined in cells that are no longer in the notebook |
| `edaic-data-preparation-train__1_` | E-DAIC | downloads archives per training participant (keeps eGeMAPS, MFCC, transcript), aggregates mean/std/median (-201 → NaN), saves 163×196 table, EDA, point-biserial correlations | ran; output = `edaic_audio_features_train.csv` (not supplied) |
| `edaic-data-preparation-test-dev` | E-DAIC | same for dev (56) and test (56) | ran |
| `edaic-audio-modeling__1_` | E-DAIC | 186-feature table: EDA (effect sizes, correlations, PCA), LR/RF/SVM on dev, train-CV, OOF thresholds, feature-set ablation | dev only; test never evaluated |
| `svm-lr__1_` | E-DAIC | 161-feature (23×7) aggregation from raw frames; tuned SVM-RBF and LogReg (+PCA variant); thresholds on dev; test once | ran |
| `random-forest__1_` | E-DAIC | **XGBoost** (despite the name) on the 161 features; early stopping on dev; threshold on dev; coefficient/importance table | ran |
| `feature-audit__1_` | E-DAIC | audit (columns, consistency across 275 files, leakage statements), re-evaluation at threshold 0.5 vs dev-selected, SVM Platt-collapse diagnosis, ANOVA-K sweep inside CV, saves joblib pipelines | ran; saved `.joblib` files not supplied |
| `bilstm__2_` | E-DAIC | pools frames by 250, BiLSTM+attention on official split | ran; 11 epochs, best dev epoch = 1 |

## Dependencies observed
pandas, numpy, matplotlib, seaborn, wordcloud, nltk (stopwords, wordnet, punkt), scikit-learn, lightgbm, xgboost, scipy, torch, tqdm, joblib. (`EDAIC_EDA__1_` also reads an audio waveform in cells whose defining code is no longer in the notebook.)

## Security / privacy scan
* No API keys, tokens, `kaggle.json`, or Colab secrets were found (`grep` for key/secret/token/password patterns).
* Absolute Kaggle paths appear throughout, including Kaggle user names in dataset paths (`entenam`, `am2008`, `vanshitasuryavanshi`). The originals are kept as-is for provenance; **decide whether you want to publish those user names** (the repository's own code contains no such paths).
* `reddit-dataset-model__2_` stores printed output containing a few verbatim Reddit posts (`df.head()`), and the E-DAIC notebooks print participant IDs and PHQ/PTSD scores. Consider clearing outputs of the notebooks in `notebooks/original/` before making the repository public (`jupyter nbconvert --clear-output`).
* The E-DAIC preparation notebooks download interview **transcripts** and audio-derived features; none of those files are included here. Check the data-use agreement before redistributing anything derived from them.
