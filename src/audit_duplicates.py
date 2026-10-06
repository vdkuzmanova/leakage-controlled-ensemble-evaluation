"""
Duplicate-record audit for the Early Stage Diabetes dataset (UCI id 529).

The dataset contains many exact duplicate rows. Under record-level splitting,
identical records land in both training and evaluation folds, which is a form
of train/test leakage. This script quantifies how much that inflates the
results by running the SAME models, grids and nested-CV design as
diabetes_pipeline.py under three protocols:

  A  "original"   - record-level splits (as in the submitted manuscript)
  B  "grouped"    - every set of identical records is one group; hold-out,
                    outer folds, inner folds and the stacking meta-learner's
                    internal CV are all group-aware (StratifiedGroupKFold),
                    so no record ever has an identical copy on the other side
  C  "dedup"      - duplicates removed (251 unique records), record-level splits

Usage:
    python audit_duplicates.py --protocols A B C
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)

from pipeline import DATASETS, RANDOM_STATE, METRICS, ROOT, model_zoo
from pipeline import pipe as make_pipeline
from pipeline import score


def _score(est, X, y, store):
    for k, v in score(est, X, y).items():
        store[k].append(v)


def summarize(scores):
    rows = [{"Model": m, **{f"{k}_mean": np.mean(v) for k, v in s.items()},
             **{f"{k}_std": np.std(v) for k, v in s.items()}} for m, s in scores.items()]
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)

warnings.filterwarnings("ignore")
ENSEMBLES = ["Voting(RF+XGB)", "Stacking(RF+XGB->LR)"]


# ---------------------------------------------------------------- audit ---
def duplicate_report(df: pd.DataFrame, target: str) -> dict:
    feats = [c for c in df.columns if c != target]
    sizes = df.groupby(list(df.columns)).size()
    conflicting = (df.groupby(feats)[target].nunique() > 1).sum()
    return {
        "n_records": int(len(df)),
        "n_unique_records": int(len(sizes)),
        "n_duplicate_records": int(df.duplicated().sum()),
        "pct_duplicate_records": round(100 * df.duplicated().mean(), 1),
        "n_patterns_with_conflicting_labels": int(conflicting),
        "positive_rate_all": round(float(df[target].mean()), 4),
        "positive_rate_unique": round(float(df.drop_duplicates()[target].mean()), 4),
        "largest_duplicate_group": int(sizes.max()),
        "group_size_distribution": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
    }


def group_ids(X: pd.DataFrame) -> np.ndarray:
    """One integer id per distinct feature pattern."""
    return X.astype(str).agg("|".join, axis=1).factorize()[0]


def overlap(X_a: pd.DataFrame, X_b: pd.DataFrame) -> int:
    """Number of rows in X_b that have an identical copy in X_a."""
    seen = set(map(tuple, X_a.values))
    return int(sum(tuple(r) in seen for r in X_b.values))


# ------------------------------------------------------- nested CV core ---
def nested_cv(X, y, groups, num, cat, grouped, outer_folds=10, inner_folds=5):
    if grouped:
        outer = StratifiedGroupKFold(outer_folds, shuffle=True, random_state=RANDOM_STATE)
        inner = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        outer = StratifiedKFold(outer_folds, shuffle=True, random_state=RANDOM_STATE)
        inner = StratifiedKFold(inner_folds, shuffle=True, random_state=RANDOM_STATE)

    zoo = model_zoo()
    metrics = METRICS
    scores = {m: {k: [] for k in metrics} for m in list(zoo) + ENSEMBLES}
    fold_overlap = []

    for i, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        g_tr = groups[tr]
        fold_overlap.append(overlap(X_tr, X_te) / len(X_te))
        fit_kw = {"groups": g_tr} if grouped else {}

        tuned = {}
        for name, (est, grid) in zoo.items():
            gs = GridSearchCV(make_pipeline(clone(est), num, cat), grid,
                              cv=inner, scoring="roc_auc", n_jobs=1)
            gs.fit(X_tr, y_tr, **fit_kw)
            tuned[name] = gs.best_estimator_
            _score(tuned[name], X_te, y_te, scores[name])

        base = [("rf", tuned["RandomForest"]), ("xgb", tuned["XGBoost"])]
        voting = VotingClassifier(base, voting="soft").fit(X_tr, y_tr)
        _score(voting, X_te, y_te, scores[ENSEMBLES[0]])

        stack_cv = (list(StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE)
                         .split(X_tr, y_tr, g_tr)) if grouped else 3)
        stacking = StackingClassifier(base, final_estimator=LogisticRegression(max_iter=1000),
                                      cv=stack_cv).fit(X_tr, y_tr)
        _score(stacking, X_te, y_te, scores[ENSEMBLES[1]])
        print(f"    outer fold {i}/{outer_folds} (test rows with identical copy in train: "
              f"{fold_overlap[-1]:.0%})", flush=True)

    return scores, float(np.mean(fold_overlap))


def holdout_auc(X_tr, y_tr, X_te, y_te, g_tr, num, cat, grouped):
    inner = (StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE) if grouped
             else StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    fit_kw = {"groups": g_tr} if grouped else {}
    out, fitted = {}, {}
    for name, (est, grid) in model_zoo().items():
        gs = GridSearchCV(make_pipeline(clone(est), num, cat), grid,
                          cv=inner, scoring="roc_auc", n_jobs=1).fit(X_tr, y_tr, **fit_kw)
        fitted[name] = gs.best_estimator_
        out[name] = roc_auc_score(y_te, fitted[name].predict_proba(X_te)[:, 1])
    return out


# ------------------------------------------------------------- protocols ---
def run_protocol(label, X, y, num, cat, grouped):
    groups = group_ids(X)
    if grouped:
        split = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE)
        tr, te = next(split.split(X, y, groups))
    else:
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    print(f"\n=== Protocol {label}: n={len(X)}, train={len(X_tr)}, hold-out={len(X_te)} ===", flush=True)
    scores, cv_overlap = nested_cv(X_tr, y_tr, groups[tr], num, cat, grouped)
    summary = summarize(scores)

    ens = summary[summary.Model.isin(ENSEMBLES)].iloc[0]["Model"]
    single = summary[~summary.Model.isin(ENSEMBLES)].iloc[0]["Model"]
    a, b = np.array(scores[ens]["roc_auc"]), np.array(scores[single]["roc_auc"])
    try:
        w, p = wilcoxon(a, b)
    except ValueError:
        w, p = np.nan, np.nan

    hold = holdout_auc(X_tr, y_tr, X_te, y_te, groups[tr], num, cat, grouped)
    return summary, {
        "protocol": label,
        "n": int(len(X)), "n_train": int(len(X_tr)), "n_holdout": int(len(X_te)),
        "holdout_rows_with_copy_in_train": overlap(X_tr, X_te),
        "mean_cv_test_rows_with_copy_in_train": round(cv_overlap, 3),
        "best_single": single, "best_ensemble": ens,
        "cv_auc_best_single": float(b.mean()), "cv_auc_best_ensemble": float(a.mean()),
        "wilcoxon_stat": float(w), "wilcoxon_p": float(p),
        "n_nonzero_fold_differences": int((a != b).sum()),
        "holdout_auc": {k: round(v, 4) for k, v in hold.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "early_stage.csv"))
    ap.add_argument("--outdir", default=str(ROOT / "results" / "duplicate_audit"))
    ap.add_argument("--protocols", nargs="+", default=["A", "B", "C"])
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = DATASETS["early_stage"]
    df = cfg["loader"](args.data)
    num, cat, target = cfg["numeric"], cfg["categorical"], cfg["target"]
    df = df[num + cat + [target]]

    report = duplicate_report(df, target)
    (outdir / "duplicate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    X, y = df[num + cat], df[target]
    dedup = df.drop_duplicates().reset_index(drop=True)
    setups = {"A": ("A_original", X, y, False),
              "B": ("B_grouped", X, y, True),
              "C": ("C_dedup", dedup[num + cat], dedup[target], False)}

    for key in args.protocols:
        label, Xp, yp, grouped = setups[key]
        summary, res = run_protocol(label, Xp, yp, num, cat, grouped)
        summary.to_csv(outdir / f"cv_{label}.csv", index=False)
        (outdir / f"summary_{label}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
