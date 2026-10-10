# Historical results from the supplied notebooks

Everything on this page was **transcribed from the stored cell outputs** of the notebooks in `notebooks/original/`.
Nothing here was re-run (the raw data were not available when this repository was built), so none of it has been
reproduced. Machine-readable copy: [`original_results.json`](original_results.json).
These numbers are **not** results of the MindText pipeline and must not be quoted as such.

## A. Reddit text (5 classes) - `reddit-dataset-model__2_.ipynb`

Protocol: legacy `clean_text` applied to all posts → random stratified 80/20 split (seed 42, **no validation split**)
→ TF-IDF (5,000 features) fitted on the training part → one fit per model → report on the 204,987 test posts.
Duplicates, "[removed]" posts and author overlap were never examined, so these scores are *not* shown to be leakage-free.

| model | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| Logistic Regression (max_iter=1000) | 0.69 | 0.51 | 0.65 |
| Multinomial NB | 0.64 | 0.32 | 0.55 |
| LinearSVC (dual=False) | 0.69 | 0.48 | 0.63 |
| Random Forest (50 trees) | 0.66 | 0.38 | 0.58 |
| LightGBM (defaults) | 0.68 | 0.48 | 0.63 |
| MLP (100 hidden) | - | - | - (never completed: no stored output; would raise `NameError` on `model.classes_`) |

Macro-F1 was computed from the notebook's printed classification reports. Per-class numbers are in the JSON; the pattern
in every model is high recall for `depression` (0.91-0.98) and very low recall for `suicidewatch` (0.01-0.15) and
`mentalhealth` (0.03-0.20). The bar chart in the notebook used hard-coded rounded values (they match the printed
reports, but the plot was not generated from the model outputs).

Class counts after filtering (1,024,932 posts): depression 613,073 · lonely 153,823 · suicidewatch 115,397 ·
mentalhealth 81,249 · anxiety 61,390. See `docs/data_card.md` for why this distribution is not trustworthy.

## B. E-DAIC acoustic features, binary `PHQ_Binary` (275 participants)

Official split: train 163 (126/37), dev 56 (44/12), test 56 (39/17); positives are 24.0 % overall. Splits are by participant,
and no participant appears in two splits (checked in the notebooks).

Test-set results as reported (positive class = `PHQ_Binary`=1, threshold chosen on dev unless stated):

| run (notebook) | features | thr | test acc | precision | recall | F1 | ROC-AUC | PR-AUC | predicts positive for |
|---|---|---|---|---|---|---|---|---|---|
| SVM-RBF, Platt probs (svm-lr) | 161 | 0.10 | 0.304 | 0.304 | 1.00 | 0.466 | 0.662 | 0.452 | **every** participant |
| LogReg C=0.01 (svm-lr) | 161 | 0.34 | 0.446 | 0.354 | 1.00 | 0.523 | 0.691 | 0.474 | 48 of 56 |
| XGBoost* (random-forest file) | 161 | 0.57 | 0.750 | 0.588 | 0.588 | **0.588** | 0.747 | 0.623 | 17 of 56 |
| SVM margin (feature-audit) | 161 | 0.60 | 0.589 | 0.421 | 0.941 | 0.582 | 0.662 | 0.452 | 38 of 56 |
| ANOVA-K50 XGBoost (feature-audit) | 161→50 | 0.12 | 0.304 | 0.304 | 1.00 | 0.466 | 0.618 | 0.444 | **every** participant |
| ANOVA-K50 LogReg (feature-audit) | 161→50 | 0.72 | 0.679 | 0.455 | 0.294 | 0.357 | 0.623 | 0.393 | 11 of 56 |
| ANOVA-K30 SVM (feature-audit) | 161→30 | 0.84 | 0.696 | 0.500 | 0.294 | 0.370 | 0.712 | 0.456 | 10 of 56 |
| BiLSTM + attention (bilstm) | 23×seq | 0.5 | 0.536 | 0.286 | 0.353 | 0.316 | - | - | 21 of 56 |

\* The notebook called `random-forest__1_.ipynb` actually trains **XGBoost**; no Random Forest result for the 161-feature
set exists in the supplied files. A Random Forest (500 trees) appears only in `edaic-audio-modeling__1_` (dev split, 186 features):
dev ROC-AUC 0.758 but F1 = 0 at the default threshold, and F1 0.407 at an out-of-fold threshold of 0.20.

Dev-only results for the 186-feature set (eGeMAPS 69 + MFCC 117; the test split is never touched in that notebook):
LogReg dev F1 0.370 (ROC-AUC 0.689) · SVM 0.333 (0.705) · RF 0.000 (0.758); 5-fold train CV F1: LR 0.214 · SVM 0.194 · RF 0.000.
Feature-set ablation (LogReg, dev): eGeMAPS-only F1 0.276 / MFCC-only 0.400 / combined 0.370.

### How to read the E-DAIC table honestly

1. **The all-positive baseline already scores F1 = 0.466 on this test set** (precision 17/56 = 0.304, recall 1.0). Three of
   the rows above are exactly that or very close to it; their "F1" says nothing about the audio features.
2. With 17 test positives the uncertainty is large. Percentile-bootstrap 95 % intervals (10,000 resamples), **computed here from the
   confusion matrices above**: XGBoost F1 0.364-0.762; all-positive rule 0.328-0.600; LogReg(0.34) 0.369-0.658; SVM-margin 0.415-0.725.
   All of these overlap. The only statement the evidence supports is that XGBoost's *threshold-free* ROC-AUC (0.747) is the highest,
   and even that comes from a single 56-person test set.
3. Dev (n=12 positives) ROC-AUCs are 0.39-0.58 for the 161-feature models, i.e. near chance, whereas test ROC-AUCs are 0.62-0.75.
   Dev and test disagree, which is what small samples look like.
4. The XGBoost run used the dev split for both early stopping and the threshold, then reported test once - an acceptable
   protocol - but re-fitting with the early-stopped number of trees gave slightly different test numbers (ROC-AUC 0.7376 vs 0.7466,
   PR-AUC 0.5971 vs 0.6232). The cause was not investigated.

## C. Issues found while auditing the notebooks (all verified from the notebook text/outputs)

| # | where | finding | handled in MindText |
|---|---|---|---|
| 1 | Reddit prep | Label from the first three characters of the file name (`ani`,`dep`,`lon`,`mhj`,`swj`); other files (e.g. `swnov20`, `mhov20`, `anxinov20`, `Swaug20 copy`, `Mheaug20`) became `unknown` and were dropped: 772,084 of 1,797,016 posts (43 %). The cell is titled "Removing noisy labels" but it removes mapping failures. The class balance is therefore partly an artefact of file naming. | `ingest.infer_label` uses explicit rules, reports unmatched files, test shows the 5 recovered file names |
| 2 | Reddit prep | `clean_text` strips digits/punctuation/emoji/apostrophes and removes NLTK stop-words, which include negations (`not`, `no`, `nor`) | off by default; `legacy_clean` for replication |
| 3 | Reddit prep | `[removed]` posts kept as `depression` training data; 3,800 posts empty after cleaning | `placeholder_texts`, `min_text_chars` |
| 4 | Reddit model | No duplicate check; no author/time information retained although the data span 2020 months | duplicate report/policy; optional group/time columns; grouped & chronological splits |
| 5 | Reddit model | Only a train/test split: no data for tuning or selection, so any choice among the five models would have to be made on the test set | train/val/test; selection on validation |
| 6 | Reddit model | MLP cell never ran; LightGBM confusion-matrix cell failed (`NameError`) and was re-run; chart uses hard-coded numbers | plots are generated from results only |
| 7 | E-DAIC | 161-feature extraction (`np.isfinite` only) keeps openSMILE's `-201` sentinel as a real value (10.28 % of eGeMAPS cells in participant 302 are -201); the 186-feature extraction replaces it with NaN. 22 of the 161 features are constant in the training data. The two feature sets are therefore not comparable. | `aggregate_frames(sentinels=…)`, default masks -201; variance filter in the pipeline |
| 8 | E-DAIC | SVM `probability=True` produced near-constant scores (all ≈ 0.228); thresholding them gives all-positive output | `svm_rbf` uses the margin; constant scores get no tuned threshold; near-all-positive thresholds raise a report warning |
| 9 | E-DAIC | Threshold chosen by F1 on 12 dev positives drifts to 0.10 / 0.12 (everything positive) | all-positive & majority baselines printed in every binary report; bootstrap CIs |
| 10 | E-DAIC | `random-forest__1_` is XGBoost; thresholds in `edaic-audio-modeling` were typed in by hand | model names come from config; thresholds computed in code |
| 11 | E-DAIC | `Gender` has four values (`female`, `female␠`, `male`, `male␠`) because of trailing spaces; `PTSD Severity` has 2 missing values | labels/strings stripped; metadata columns excluded from features by config |
| 12 | E-DAIC | `PHQ_Binary` is derived from `PHQ_Score` (dev row 320: score 11 → 0; test rows 602/604: scores 13/12 → 1). The cut-off is therefore not obviously 10 - **confirm in the E-DAIC documentation** | `PHQ_Score` is never written to the feature table |
| 13 | BiLSTM | Best dev F1 at epoch 1 (essentially untrained); a "classical F1=0.632 baseline" is cited but not found in any supplied output | not ported (needs PyTorch; unavailable here) - notebook kept |
| 14 | all | Hard-coded `/kaggle/...` paths; Kaggle user names embedded in dataset paths | paths come from config/CLI |
