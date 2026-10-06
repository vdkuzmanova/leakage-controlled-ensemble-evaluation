# Leakage-Controlled Evaluation of Ensemble Superiority Claims in Diabetes Risk Prediction — code and results

Reference implementation for the manuscript by V. D. Kuzmanova (UCTM, Sofia).
The framework **F = (P, V, S, E)** evaluates tabular medical prediction models under:

- **P – fold-safe preprocessing.** Imputation, scaling/encoding and SMOTE are fit only on each training fold (imblearn `Pipeline`).
- **V – group-aware repeated nested CV.** 10 repeats × 10 outer folds with 5-fold inner `GridSearchCV`. Records with identical feature+label patterns form one group, so no record has an identical copy on the opposite side of any split. This covers the hold-out split, the outer and inner folds, and the stacking meta-learner's internal CV.
- **S – statistical comparison.** Two tests are run on the 100 paired outer-fold ROC-AUC differences:
  - the corrected resampled t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004);
  - the Bayesian correlated t-test with a region of practical equivalence of ±0.01 ROC-AUC (Corani & Benavoli, 2015; Benavoli et al., 2017).
- **E – explanation.** SHAP is computed for the model chosen by a pre-specified rule: the best single model by mean CV ROC-AUC among models with an exact explainer (`LinearExplainer` for logistic regression, `TreeExplainer` for tree models). A cross-model agreement check reports Kendall τ between the SHAP feature rankings of LR, RF and XGBoost.

## Repository structure

```
src/pipeline.py            framework: data loading, P, V, S, E, cross-dataset summary
src/audit_duplicates.py    duplicate-record audit (Early Stage dataset), three protocols
data/                      pima.csv, early_stage.csv (UCI)
results/<dataset>/         folds/repeat_*.csv (every outer fold), cv_summary.csv,
                           statistics.json, holdout.csv, shap_summary.json, ...
results/duplicate_audit/   record-level vs. grouped vs. deduplicated evaluation
figures/<dataset>/         hold-out ROC curves, SHAP beeswarm, SHAP agreement plot
```

## Reproducing the results

```bash
pip install -r requirements-lock.txt   # exact versions used for the manuscript (Python 3.12.3)
cd src
python pipeline.py --dataset pima        --stage cv --repeats 0 1 2 3 4 5 6 7 8 9
python pipeline.py --dataset pima        --stage final
python pipeline.py --dataset early_stage --stage cv --repeats 0 1 2 3 4 5 6 7 8 9
python pipeline.py --dataset early_stage --stage final
python pipeline.py --stage compare
python audit_duplicates.py --protocols A B C
```

Each repeat is written to disk as soon as it finishes, so a long run can be split across sessions. On a single CPU core one repeat takes about 2 minutes (Early Stage) or 3 minutes (PIDD).

## Note on the Early Stage dataset

269 of its 520 records (51.7%) are exact duplicates of another record, leaving 251 distinct records. With record-level splitting, about two thirds of each test fold has an identical copy in the training data. All splits on this dataset are therefore group-aware. `audit_duplicates.py` quantifies the effect of this choice.

## Reproducibility notes

All random components are seeded: hold-out split, SMOTE and model initialisation use seed 42; outer, inner and stacking CV splitters use seed 42 + repeat index; the 50 ablation splits use seeds 1000-1049. Per-fold metrics and selected hyperparameters are saved in `results/<dataset>/folds/`. Fold assignments are not stored but are regenerated deterministically from the seeds; per-record predictions are not stored. Exact numerical agreement may depend on library versions and platform; `requirements-lock.txt` lists the versions used.

## Revision analyses (v6)

- `results/<dataset>_hetero/`, `figures/<dataset>_hetero/`: 10x10 nested CV including two heterogeneous five-model ensembles (secondary analysis): `python pipeline.py --dataset <d> --stage cv --repeats 0 ... 9 --tag _hetero`, then `--stage final --tag _hetero`. The eight original models reproduce the primary results exactly.
- `results/ablation_v2/`: L2-L4 re-run with additional scoring on original (non-synthetic) test records (`ABLATION_DIR=ablation_v2 python ablation.py ...`); `python ablation_summary_v2.py` builds the summary used in Table 4, including the like-for-like ensemble-versus-own-base-learner comparison.
