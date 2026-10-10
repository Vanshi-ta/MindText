# Limitations and ethical considerations

**MindText is a machine-learning research and teaching project. It is not a diagnostic tool and must not be used to assess, screen, label, or make decisions about any person.**

1. **Labels are not diagnoses.** Reddit labels (as inferred) reflect where text was posted. E-DAIC's label is a questionnaire-derived cut-off collected in a research interview. Neither is a clinical assessment.
2. **Non-representative data.** Reddit users are self-selected; posts are public, informal, English-language, and from 2020. The E-DAIC participants are a few hundred people interviewed by a virtual agent. Neither supports claims about the general population, other languages, other platforms, or other demographics. No fairness/subgroup analysis was performed.
3. **Tiny acoustic test set.** 56 participants, 17 positives: one participant moves F1 by several points; intervals are very wide (see `original_results.md`). Differences between models in the historical results are not statistically supported.
4. **Possible leakage in the historical Reddit scores:** random post-level split, no duplicate handling, no author information. MindText adds duplicate handling, grouped/chronological splits (when columns exist), and leakage reports, but it cannot detect near-duplicates or authors that are not identified in the data.
5. **Label-source artefacts.** Class proportions in the Reddit data are partly determined by file-naming (original filter) and by community size.
6. **Class imbalance.** Accuracy rewards predicting the largest class; report macro-F1, per-class recall and the baselines the tool prints.
7. **Explanations are associations.** Top-term lists and feature importances are properties of one fitted model. They do not explain why a person feels or behaves a certain way and must not be shown as "signs of depression".
8. **No new real-data results.** The environment in which this repository was assembled had no access to the datasets, so every number produced by MindText code in this repository comes from synthetic data and is meaningless. Historical numbers come from your notebooks and are labelled as such.
9. **Model files are pickles.** Load only files you created or trust (`persistence.py`); the checksum sidecar detects accidental corruption, not tampering by someone who can write both files.
10. **Privacy.** Error-analysis excerpts are redacted (URLs, e-mails, handles, long digit runs) and truncated, can be switched off (`error_analysis.include_text: false`), and cite only a record id. Redaction is heuristic and not a guarantee of anonymity. Author names, if present, are replaced by a salted hash (pseudonymisation, not anonymisation). Do not publish posts, transcripts or participant-level outputs.
11. **Do not build user-facing screening on this.** If you or someone you know may be in crisis, contact local emergency services or a crisis line; the demo app says so and never outputs risk levels.
