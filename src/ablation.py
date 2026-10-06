"""
Ablation of evaluation bias: how much do common protocol shortcuts inflate
reported performance, and do they create an apparent ensemble advantage?

Every protocol evaluates the same eight models on a single stratified 80/20
split, repeated over S random seeds, mimicking a typical single-split study.
Protocols add one bias source at a time (cumulative):

  L0  clean       preprocessing and SMOTE fit on the training part only
  L1  +prep       imputation, scaling and encoding fit on the full dataset
  L2  +SMOTE      L1, and SMOTE applied to the full dataset before splitting
  L3  +tuning     L2, and hyperparameters tuned by CV on the full (oversampled)
                  dataset, then evaluated on splits of that same data
  L4  +cleaning   L3, and outlier rejection (1.5 IQR on numeric features) and
                  correlation-based feature selection on the full dataset

Early Stage dataset only: L0 is run twice, with group-aware splits
(L0-group: no record has an identical copy on the other side) and with
record-level splits (L0). L1-L3 use record-level splits, as in the literature.

In L0-L2 the hyperparameters are fixed to the modal values selected by the
clean nested CV (results/<dataset>/hyperparameters_modal.json), so the only
difference between these protocols is where preprocessing and SMOTE are fit.

  python ablation.py --dataset pima --protocol L0 --seeds 50
  python ablation.py --analyse
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)

from pipeline import (DATASETS, ENSEMBLES, RANDOM_STATE, ROOT, load,
                      model_zoo, preprocessor, score)

warnings.filterwarnings("ignore")
PROTOCOLS = {
    #          prep_leak smote_leak tune_leak grouped
    "L0-group": (False, False, False, True),
    "L0":       (False, False, False, False),
    "L1":       (True,  False, False, False),
    "L2":       (True,  True,  False, False),
    "L3":       (True,  True,  True,  False),
    "L4":       (True,  True,  True,  False),
}
ORDER = ["L0-group", "L0", "L1", "L2", "L3", "L4"]
# L4 = L3 + outlier rejection (1.5 IQR on numeric features) and correlation-based
# feature selection (|Pearson r| with the label >= FS_THRESHOLD), both on the full
# oversampled dataset before splitting, as described in several published pipelines.
FS_THRESHOLD = 0.10
import os
OUT = ROOT / "results" / os.environ.get("ABLATION_DIR", "ablation")


def build(clf, num, cat, prep_leak, smote_leak):
    steps = [] if prep_leak else [("prep", preprocessor(num, cat))]
    if not smote_leak:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def models(params, num, cat, prep_leak, smote_leak):
    out = {}
    for name, (est, _) in model_zoo().items():
        p = build(clone(est), num, cat, prep_leak, smote_leak)
        p.set_params(**params[name])
        out[name] = p
    base = [("rf", out["RandomForest"]), ("xgb", out["XGBoost"])]
    out[ENSEMBLES[0]] = VotingClassifier(base, voting="soft")
    out[ENSEMBLES[1]] = StackingClassifier(
        base, final_estimator=LogisticRegression(max_iter=1000), cv=3)
    return out


def run(key, protocol, n_seeds):
    prep_leak, smote_leak, tune_leak, grouped = PROTOCOLS[protocol]
    cfg, X, y, groups = load(key)
    num, cat = cfg["numeric"], cfg["categorical"]

    miss = X[cfg["numeric"]].isna().to_numpy()      # originally missing entries
    if prep_leak:                                   # L1+: fit on everything
        X = pd.DataFrame(preprocessor(num, cat).fit_transform(X))
        X.columns = [str(c) for c in X.columns]
        num, cat = list(X.columns), []
    n_orig = len(y)
    if smote_leak:                                  # L2+: oversample everything
        X, y = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
        groups = np.arange(len(y))
    is_orig = np.arange(len(y)) < n_orig            # SMOTE appends synthetic rows
    info = {}
    if protocol == "L4":                            # L4: clean/select on everything
        X, y = X.reset_index(drop=True), pd.Series(np.asarray(y)).reset_index(drop=True)
        n_num = len(cfg["numeric"])                 # numeric columns come first
        numeric_cols = list(X.columns[:n_num])
        # SMOTE appends synthetic rows after the originals; imputed entries are
        # excluded from the fences and never flagged (outliers are judged on
        # observed values, as when rejection precedes imputation)
        was_missing = np.zeros((len(X), n_num), dtype=bool)
        was_missing[:len(miss)] = miss
        obs = X[numeric_cols].mask(was_missing)
        q1, q3 = obs.quantile(0.25), obs.quantile(0.75)
        iqr = q3 - q1
        flag = ((obs < q1 - 1.5 * iqr) | (obs > q3 + 1.5 * iqr)).fillna(False)
        keep = ~flag.any(axis=1)
        info["n_before_outlier_rejection"] = int(len(X))
        X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        is_orig = is_orig[keep.to_numpy()]
        info["n_after_outlier_rejection"] = int(len(X))
        feat_names = cfg["numeric"] + cfg["categorical"]
        r = X.apply(lambda c: np.corrcoef(c, y)[0, 1] if c.std() > 0 else 0.0)
        selected = [c for c in X.columns if abs(r[c]) >= FS_THRESHOLD]
        info["features_kept"] = [feat_names[int(c)] for c in selected]
        info["features_dropped"] = [feat_names[int(c)] for c in X.columns if c not in selected]
        X = X[selected]; num = selected
        groups = np.arange(len(y))

    params = {m: v["params"] for m, v in json.loads(
        (ROOT / "results" / key / "hyperparameters_modal.json").read_text()).items()}
    if tune_leak:                                   # L3: tune on everything
        cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
        for name, (est, grid) in model_zoo().items():
            gs = GridSearchCV(build(clone(est), num, cat, prep_leak, smote_leak), grid,
                              cv=cv, scoring="roc_auc", n_jobs=1).fit(X, y)
            params[name] = gs.best_params_

    rows = []
    for s in range(n_seeds):
        seed = 1000 + s
        if grouped:
            tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                          .split(X, y, groups))
        else:
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                                      stratify=y, random_state=seed)
        for name, est in models(params, num, cat, prep_leak, smote_leak).items():
            est.fit(X.iloc[tr], y.iloc[tr])
            row = {"dataset": key, "protocol": protocol, "seed": seed,
                   "model": name, **score(est, X.iloc[te], y.iloc[te])}
            te_o = te[is_orig[te]]                  # test records that are not synthetic
            row.update({"orig_" + k: v for k, v in score(est, X.iloc[te_o], y.iloc[te_o]).items()})
            row.update({"n_test": len(te), "n_test_orig": len(te_o),
                        "test_pos_rate": float(y.iloc[te].mean()), "test_pos_rate_orig": float(y.iloc[te_o].mean())})
            rows.append(row)
        print(f"  [{key} {protocol}] seed {s + 1}/{n_seeds}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{key}_{protocol}.csv", index=False)
    (OUT / f"{key}_{protocol}_params.json").write_text(
        json.dumps({"params": params, **info}, indent=2, default=str))


# ------------------------------------------------------------ analysis ---
def analyse():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("*_L*.csv"))])
    df["is_ens"] = df["model"].isin(ENSEMBLES)
    summary, lines = [], ["# Ablation summary\n"]

    for key in DATASETS:
        d = df[df.dataset == key]
        if d.empty:
            continue
        ref = d[d.protocol == ("L0-group" if key == "early_stage" else "L0")]
        ref_mean = ref.groupby("model")[["accuracy", "roc_auc"]].mean()
        lines.append(f"\n## {DATASETS[key]['name']}")
        for prot in [p for p in ORDER if p in d.protocol.unique()]:
            p = d[d.protocol == prot]
            m = p.groupby("model")[["accuracy", "roc_auc"]].mean()
            # per-seed view: what a single-split study would have reported
            per_seed = []
            for _, g in p.groupby("seed"):
                best_s = g[~g.is_ens].accuracy.max(); best_e = g[g.is_ens].accuracy.max()
                per_seed.append({"top_is_ensemble": bool(best_e > best_s),
                                 "top_tied": bool(best_e == best_s),
                                 "ens_gap_acc": best_e - best_s,
                                 "best_acc": g.accuracy.max(),
                                 "best_auc": g.roc_auc.max()})
            ps = pd.DataFrame(per_seed)
            row = {"dataset": key, "protocol": prot, "n_seeds": len(ps),
                   "mean_acc_all_models": m.accuracy.mean(),
                   "mean_auc_all_models": m.roc_auc.mean(),
                   "inflation_acc": (m.accuracy - ref_mean.accuracy).mean(),
                   "inflation_auc": (m.roc_auc - ref_mean.roc_auc).mean(),
                   "inflation_acc_RF": m.accuracy["RandomForest"] - ref_mean.accuracy["RandomForest"],
                   "inflation_acc_LR": m.accuracy["LogisticRegression"] - ref_mean.accuracy["LogisticRegression"],
                   "best_model_mean_acc": m.accuracy.max(),
                   "best_model": m.accuracy.idxmax(),
                   "reported_acc_median_seed": ps.best_acc.median(),
                   "reported_acc_max_seed": ps.best_acc.max(),
                   "reported_auc_max_seed": ps.best_auc.max(),
                   "pct_seeds_top_is_ensemble": 100 * ps.top_is_ensemble.mean(),
                   "mean_ens_gap_acc": ps.ens_gap_acc.mean(),
                   "pct_seeds_ens_gap_ge_1pt": 100 * (ps.ens_gap_acc >= 0.01).mean(),
                   "pct_seeds_tied": 100 * ps.top_tied.mean(),
                   "pct_seeds_single_gap_ge_1pt": 100 * (ps.ens_gap_acc <= -0.01).mean()}
            summary.append(row)
            lines.append(
                f"- {prot}: mean acc {row['mean_acc_all_models']:.3f} "
                f"(inflation {row['inflation_acc']:+.3f}; RF {row['inflation_acc_RF']:+.3f}, "
                f"LR {row['inflation_acc_LR']:+.3f}), best reported acc median/max over seeds "
                f"{row['reported_acc_median_seed']:.3f}/{row['reported_acc_max_seed']:.3f}, "
                f"ensemble strictly on top in {row['pct_seeds_top_is_ensemble']:.0f}% of seeds "
                f"(tied {row['pct_seeds_tied']:.0f}%), ensemble ≥1 pt ahead in "
                f"{row['pct_seeds_ens_gap_ge_1pt']:.0f}%, single model ≥1 pt ahead in "
                f"{row['pct_seeds_single_gap_ge_1pt']:.0f}%")
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "ablation_summary.csv", index=False)
    (OUT / "ablation_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    figures(df)


def figures(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, key in zip(axes, DATASETS):
        d = df[df.dataset == key]
        prots = [p for p in ORDER if p in d.protocol.unique()]
        best = [d[d.protocol == p].groupby("seed").accuracy.max().values for p in prots]
        ax.boxplot(best, labels=prots, showfliers=True)
        ax.set_title(DATASETS[key]["short"], fontsize=10)
        ax.set_ylabel("Best accuracy reported on a single split")
        ax.set_xlabel("Protocol (cumulative bias sources)")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_dir = ROOT / "figures" / "ablation"; fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "reported_accuracy_by_protocol.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--protocol", choices=ORDER)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        return analyse()
    if a.dataset == "pima" and a.protocol == "L0-group":
        raise SystemExit("PIDD has no duplicates; L0-group is identical to L0.")
    run(a.dataset, a.protocol, a.seeds)


if __name__ == "__main__":
    main()
