# Methodology

## Pipeline
```
raw files ──(prepare-reddit | build-acoustic-features)──► tidy CSV
   │                                                         
   ▼                                                         
load_table → validate_and_clean (schema, labels, missing/placeholder text, duplicates, QualityReport)
   ▼
make_split  (stratified | grouped | chronological | predefined)  → leakage_report
   ▼
for each model:  Pipeline[ TextCleaner → TF-IDF | impute → variance → scale → (select) ] → classifier
                 tune by CV on TRAIN only → (binary) choose threshold on VALIDATION
   ▼
select model on VALIDATION → evaluate ALL models on TEST once → bootstrap CIs, baselines
   ▼
report.json · figures · predictions · error_analysis.json · model.joblib (+ .meta.json)
```

## Leakage controls
| risk | control | where tested |
|---|---|---|
| vocabulary/IDF, imputation, scaling, feature selection fitted on held-out data | all inside one sklearn `Pipeline`; `fit` only sees train (or CV-train folds) | `test_vectorizer_vocabulary_…`, `test_imputer_and_scaler_…` |
| tuning on test | `GridSearchCV` on the train split; selection & threshold on validation; config validation refuses to select among models without a validation split | `test_selection_uses_validation_not_test`, `test_selection_without_validation_split_rejected`, `test_tabular_run_threshold_chosen_on_validation` |
| overlapping rows | disjoint index/record-id check (hard error) | `test_partitions_are_disjoint…` |
| repeated authors / threads | `grouped` strategy (hard error if groups overlap); overlap reported for other strategies | `test_grouped_*` |
| time drift | `chronological` strategy | `test_chronological_*` |
| duplicate texts | policy before splitting (`drop_exact`, `drop_text_conflicts`), cross-split duplicate count in the report | `test_dedup_policies`, `test_cross_split_duplicate_texts_are_flagged` |
| target leakage via metadata | `exclude_columns`, non-numeric columns auto-excluded, `PHQ_Score` never exported | `test_metadata_columns_never_become_features` |
| resampling | none is used (class imbalance is handled by `class_weight`/`scale_pos_weight` inside training folds only) | - |

Near-duplicates (paraphrases, cross-posts with edits) and unidentified repeat authors are **not** detectable here; see `limitations.md`.

## Text preprocessing (defaults)
Lower-case, HTML unescape/strip, URLs → `urltoken`, whitespace collapse. **Kept**: negations, punctuation characters, digits, emoji in the cleaned string. Caveat: scikit-learn's default tokeniser (`\b\w\w+\b`) still discards punctuation/emoji and one-letter words when it builds the TF-IDF vocabulary; set `tfidf.token_pattern: "(?u)\\b\\w+\\b|[^\\w\\s]"` to keep them. Neither choice has been evaluated on the real data; run the ablation (stop-words on/off, punctuation tokens on/off, lemmatisation) before claiming one is better. Features: TF-IDF uni+bi-grams, `min_df=2`, 20k features, sublinear TF.

## Acoustic preprocessing
Frames → per-participant statistics (mean, std, median, min, max, p25, p75), with `-201` masked as missing by default → median imputation → variance filter → standardisation (for scale-sensitive models) → optional ANOVA `SelectKBest` (inside the pipeline, hence inside each CV fold).

## Models
`majority`, `nb` (text), `logreg`, `linearsvc`, `svm_rbf`, `rf`, `xgboost`, `lightgbm` (the last two optional extras). Each configured model carries a `provenance` tag: `original` = trained in your notebooks, `new` = added here (currently only `logreg_balanced`, a class-weighted variant motivated by the near-zero recall of minority classes in the historical results). The historical BiLSTM is not ported.

## Evaluation
Accuracy, balanced accuracy, macro/weighted P/R/F1, per-class P/R/F1, MCC, confusion matrix, classification report; binary: positive-class P/R/F1, specificity, TP/FP/TN/FN, ROC-AUC, PR-AUC (zero-division → 0). Accuracy can reach 95 % with a model that finds no positives (`test_accuracy_can_hide_a_useless_model`), so selection defaults to macro-F1. Binary reports also print the all-positive and majority baselines, and test metrics carry percentile-bootstrap 95 % CIs (resampling test records; with small n they are wide). SVM scores are margins, not probabilities, and are labelled `decision`.

## Reproducibility
Seeds from `train.seed`; sklearn `random_state`s; run id = UTC timestamp + task + config hash; the report stores config, data SHA-256, split assignments (`split_assignments.csv.gz`), package versions, model checksum. Bit-for-bit reproduction across different library versions is not guaranteed.
