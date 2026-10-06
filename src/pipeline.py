"""
Leakage-safe evaluation framework F = (P, V, S, E) for tabular medical ML.

P  fold-safe preprocessing: imputation, scaling/encoding and SMOTE inside an
   imblearn Pipeline, fit only on each training fold.
V  group-aware repeated nested cross-validation: R repeats x K outer folds,
   inner GridSearchCV. Records with identical feature+label patterns form one
   group, so no record ever has an identical copy on the other side of a
   split (hold-out, outer fold, inner fold, stacking meta-learner CV).
S  statistical comparison on the R*K outer-fold ROC-AUC differences:
   corrected resampled t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004)
   and Bayesian correlated t-test with a region of practical equivalence
   (Corani & Benavoli 2015; Benavoli et al. 2017).
E  SHAP explanation of the selected model on the held-out set, plus a
   cross-model agreement check (Kendall tau between feature rankings of
   LR, RF and XGBoost).

Pre-specified selection rule for E: the best single model by mean CV ROC-AUC
among models with an exact SHAP explainer (LinearExplainer: LogisticRegression;
TreeExplainer: DecisionTree, RandomForest, XGBoost).

Stages (each repeat is checkpointed to disk so long runs can be resumed):
  python pipeline.py --dataset pima --stage cv --repeats 0 1 2 3 4
  python pipeline.py --dataset pima --stage final
  python pipeline.py --stage compare
"""

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42          # model initialisation seed, fixed for all runs
OUTER_FOLDS, INNER_FOLDS, STACK_FOLDS = 10, 5, 3
HOLDOUT_FOLDS = 5          # one of 5 group-stratified folds = 20% hold-out
ROPE = 0.01                # region of practical equivalence, ROC-AUC units
METRICS = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]
ENSEMBLES = ["Voting(RF+XGB)", "Stacking(RF+XGB->LR)", "Voting(all)", "Stacking(all->LR)"]
# heterogeneous ensembles combine five model families (Decision Tree excluded, as it is dominated by Random Forest)
HETERO_BASE = ["LogisticRegression", "SVM", "NeuralNetwork", "RandomForest", "XGBoost"]
EXACT_SHAP = {"LogisticRegression": "linear", "DecisionTree": "tree",
              "RandomForest": "tree", "XGBoost": "tree"}
AGREEMENT_MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ data ---
PIMA_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


def load_pima(path):
    first = Path(path).read_text().splitlines()[0]
    has_header = any(c.isalpha() for c in first)
    df = pd.read_csv(path) if has_header else pd.read_csv(path, header=None, names=PIMA_COLS)
    df = df[PIMA_COLS].copy()
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    return df


def load_early_stage(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["target"] = (df["class"].str.strip().str.lower() == "positive").astype(int)
    return df.drop(columns=["class"])


DATASETS = {
    "pima": {
        "name": "Pima Indians Diabetes Dataset", "short": "PIDD",
        "file": "data/pima.csv", "loader": load_pima, "target": "Outcome",
        "numeric": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "categorical": [],
    },
    "early_stage": {
        "name": "Early Stage Diabetes Risk Prediction Dataset", "short": "Early Stage",
        "file": "data/early_stage.csv", "loader": load_early_stage, "target": "target",
        "numeric": ["Age"],
        "categorical": ["Gender", "Polyuria", "Polydipsia", "sudden weight loss",
                        "weakness", "Polyphagia", "Genital thrush", "visual blurring",
                        "Itching", "Irritability", "delayed healing", "partial paresis",
                        "muscle stiffness", "Alopecia", "Obesity"],
    },
}


DEDUP = False   # set by --dedup: drop exact duplicate records (features + label) before any split


def load(key):
    cfg = DATASETS[key]
    df = cfg["loader"](ROOT / cfg["file"])
    feats = cfg["numeric"] + cfg["categorical"]
    if DEDUP:
        df = df.drop_duplicates(subset=feats + [cfg["target"]]).reset_index(drop=True)
    X, y = df[feats].reset_index(drop=True), df[cfg["target"]].reset_index(drop=True)
    # identical feature+label patterns share a group id
    groups = pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=False).factorize()[0]
    return cfg, X, y, groups


def audit(X, y, groups):
    miss = X.isna().mean().round(4)
    sizes = pd.Series(groups).value_counts()
    return {
        "n_records": int(len(X)), "n_groups": int(sizes.size),
        "n_duplicate_records": int(len(X) - sizes.size),
        "pct_duplicate_records": round(100 * (len(X) - sizes.size) / len(X), 1),
        "largest_group": int(sizes.max()),
        "positive_rate": round(float(y.mean()), 4),
        "missing_fraction": {k: float(v) for k, v in miss[miss > 0].items()},
    }


def holdout_split(y, groups):
    sgkf = StratifiedGroupKFold(HOLDOUT_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return next(sgkf.split(np.zeros(len(y)), y, groups))


# --------------------------------------------------------- models / P ---
def preprocessor(num, cat):
    steps = []
    if num:
        steps.append(("num", SkPipeline([("impute", SimpleImputer(strategy="median")),
                                         ("scale", StandardScaler())]), num))
    if cat:
        steps.append(("cat", SkPipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                         ("encode", OrdinalEncoder())]), cat))
    return ColumnTransformer(steps)


def pipe(est, num, cat):
    return ImbPipeline([("prep", preprocessor(num, cat)),
                        ("smote", SMOTE(random_state=RANDOM_STATE)),
                        ("clf", est)])


def model_zoo():
    return {
        "LogisticRegression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                               {"clf__C": [0.1, 1, 10]}),
        "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                         {"clf__max_depth": [5, 7, None], "clf__min_samples_leaf": [1, 5]}),
        "RandomForest": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
                         {"clf__max_depth": [5, 10], "clf__min_samples_leaf": [1, 3]}),
        "SVM": (SVC(kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE),
                {"clf__C": [1, 10]}),
        "XGBoost": (XGBClassifier(n_estimators=200, eval_metric="logloss",
                                  random_state=RANDOM_STATE, n_jobs=1),
                    {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]}),
        "NeuralNetwork": (MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000,
                                        random_state=RANDOM_STATE),
                          {"clf__alpha": [1e-4, 1e-3]}),
    }


def tune_all(X, y, g, num, cat, seed):
    """Inner-loop tuning of every single model, then the two ensembles."""
    inner = StratifiedGroupKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    fitted, params = {}, {}
    for name, (est, grid) in model_zoo().items():
        gs = GridSearchCV(pipe(clone(est), num, cat), grid, cv=inner,
                          scoring="roc_auc", n_jobs=1)
        gs.fit(X, y, groups=g)
        fitted[name], params[name] = gs.best_estimator_, gs.best_params_
    base = [("rf", fitted["RandomForest"]), ("xgb", fitted["XGBoost"])]
    fitted[ENSEMBLES[0]] = VotingClassifier(base, voting="soft").fit(X, y)
    stack_cv = list(StratifiedGroupKFold(STACK_FOLDS, shuffle=True, random_state=seed)
                    .split(X, y, g))
    fitted[ENSEMBLES[1]] = StackingClassifier(
        base, final_estimator=LogisticRegression(max_iter=1000), cv=stack_cv).fit(X, y)
    hbase = [(n, fitted[n]) for n in HETERO_BASE]
    fitted[ENSEMBLES[2]] = VotingClassifier(hbase, voting="soft").fit(X, y)
    fitted[ENSEMBLES[3]] = StackingClassifier(
        hbase, final_estimator=LogisticRegression(max_iter=1000), cv=stack_cv).fit(X, y)
    return fitted, params


def score(est, X, y):
    pred, prob = est.predict(X), est.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, prob)}


# ------------------------------------------------------------------- V ---
def run_repeat(key, r, outdir):
    cfg, X, y, groups = load(key)
    tr, _ = holdout_split(y, groups)
    X, y, groups = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), groups[tr]
    seed = RANDOM_STATE + r
    outer = StratifiedGroupKFold(OUTER_FOLDS, shuffle=True, random_state=seed)
    records = []
    for k, (a, b) in enumerate(outer.split(X, y, groups)):
        fitted, params = tune_all(X.iloc[a], y.iloc[a], groups[a],
                                  cfg["numeric"], cfg["categorical"], seed)
        for name, est in fitted.items():
            rec = {"repeat": r, "fold": k, "model": name,
                   "n_train": len(a), "n_test": len(b), **score(est, X.iloc[b], y.iloc[b])}
            if name in params:
                rec["params"] = json.dumps(params[name], sort_keys=True, default=str)
            records.append(rec)
        print(f"  [{key}] repeat {r} fold {k + 1}/{OUTER_FOLDS}", flush=True)
    folds_dir = outdir / "folds"; folds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(folds_dir / f"repeat_{r}.csv", index=False)


# ------------------------------------------------------------------- S ---
def compare(a, b, n_train, n_test, rope=ROPE):
    """a, b: per-fold ROC-AUC of two models over R*K outer folds (paired)."""
    d = np.asarray(a) - np.asarray(b)
    n, mean, var = len(d), d.mean(), d.var(ddof=1)
    scale = np.sqrt((1 / n + n_test / n_train) * var) if var > 0 else 1e-12
    t = mean / scale
    post = stats.t(df=n - 1, loc=mean, scale=scale)
    return {"mean_diff": float(mean), "sd_diff": float(np.sqrt(var)), "n_folds": int(n),
            "corrected_t": float(t),
            "corrected_p": float(2 * stats.t.sf(abs(t), df=n - 1)),
            "ci95": [float(x) for x in post.interval(0.95)],
            "rope": rope,
            "p_a_better": float(post.sf(rope)),
            "p_equivalent": float(post.cdf(rope) - post.cdf(-rope)),
            "p_b_better": float(post.cdf(-rope))}


def statistics(folds):
    auc = folds.pivot_table(index=["repeat", "fold"], columns="model", values="roc_auc")
    n_train, n_test = folds["n_train"].mean(), folds["n_test"].mean()
    means = auc.mean().sort_values(ascending=False)
    singles = [m for m in means.index if m not in ENSEMBLES]
    ens = [m for m in means.index if m in ENSEMBLES]
    best_single, best_ens = singles[0], ens[0]
    out = {"best_single": best_single, "best_ensemble": best_ens,
           "comparisons": {}}
    pairs = [(best_ens, best_single)] + [(e, s) for e in ENSEMBLES[:2]
                                         for s in ("RandomForest", "XGBoost")] \
            + [(e, s) for e in ENSEMBLES[2:] for s in HETERO_BASE]
    for a, b in dict.fromkeys(pairs):
        out["comparisons"][f"{a} vs {b}"] = compare(auc[a], auc[b], n_train, n_test)
    return out, means


# ------------------------------------------------------------------- E ---
def shap_values(name, pipeline_, X_bg, X_eval, feats):
    prep, clf = pipeline_.named_steps["prep"], pipeline_.named_steps["clf"]
    Xb = pd.DataFrame(prep.transform(X_bg), columns=feats)
    Xe = pd.DataFrame(prep.transform(X_eval), columns=feats)
    if EXACT_SHAP[name] == "linear":
        expl = shap.LinearExplainer(clf, Xb)
    else:
        expl = shap.TreeExplainer(clf)
    sv = expl(Xe)
    vals = sv.values[..., 1] if sv.values.ndim == 3 else sv.values
    base = sv.base_values[..., 1] if np.ndim(sv.base_values) == 2 else sv.base_values
    return shap.Explanation(vals, base_values=base, data=Xe.values, feature_names=feats)


def final_stage(key, outdir, figdir):
    cfg, X, y, groups = load(key)
    feats = cfg["numeric"] + cfg["categorical"]
    tr, te = holdout_split(y, groups)
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    info = audit(X, y, groups)
    info.update({"dataset": cfg["name"], "n_train": int(len(tr)), "n_holdout": int(len(te)),
                 "holdout_records_with_identical_copy_in_train":
                     int(np.isin(groups[te], groups[tr]).sum())})

    folds = pd.concat([pd.read_csv(p) for p in sorted((outdir / "folds").glob("repeat_*.csv"))])
    info["repeats"] = int(folds["repeat"].nunique())
    info["outer_folds"] = OUTER_FOLDS

    # CV summary table
    summ = folds.groupby("model")[METRICS].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.sort_values("roc_auc_mean", ascending=False)
    summ.to_csv(outdir / "cv_summary.csv")

    # selected hyperparameters (modal choice over all outer folds)
    modal = {m: Counter(g["params"].dropna()).most_common(1)[0]
             for m, g in folds.groupby("model") if g["params"].notna().any()}
    (outdir / "hyperparameters_modal.json").write_text(json.dumps(
        {m: {"params": json.loads(p), "chosen_in_folds": c, "of": int(len(folds) // folds.model.nunique())}
         for m, (p, c) in modal.items()}, indent=2))

    # S
    sig, means = statistics(folds)
    (outdir / "statistics.json").write_text(json.dumps(sig, indent=2))

    # refit on full training part, evaluate once on hold-out
    fitted, _ = tune_all(X_tr, y_tr, groups[tr], cfg["numeric"], cfg["categorical"], RANDOM_STATE)
    hold = pd.DataFrame({m: score(e, X_te, y_te) for m, e in fitted.items()}).T
    hold.loc[means.index].to_csv(outdir / "holdout.csv")

    plt.figure(figsize=(6.5, 5.5))
    for m in means.index:
        fpr, tpr, _ = roc_curve(y_te, fitted[m].predict_proba(X_te)[:, 1])
        plt.plot(fpr, tpr, lw=1.3, label=f"{m} (AUC={hold.loc[m, 'roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"Hold-out ROC — {cfg['short']}", fontsize=10); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(figdir / "roc_holdout.png", dpi=200); plt.close()

    # E: pre-specified selection rule
    selected = next(m for m in means.index if m in EXACT_SHAP)
    explained = {}
    for m in dict.fromkeys([selected] + AGREEMENT_MODELS):
        explained[m] = shap_values(m, fitted[m], X_tr, X_te, feats)
    imp = pd.DataFrame({m: np.abs(e.values).mean(0) for m, e in explained.items()}, index=feats)
    imp.sort_values(selected, ascending=False).to_csv(outdir / "shap_importance.csv")

    agree = {}
    for i, a in enumerate(AGREEMENT_MODELS):
        for b in AGREEMENT_MODELS[i + 1:]:
            tau, p = stats.kendalltau(imp[a], imp[b])
            top_a = set(imp[a].nlargest(3).index); top_b = set(imp[b].nlargest(3).index)
            agree[f"{a} vs {b}"] = {"kendall_tau": float(tau), "p": float(p),
                                    "top3_overlap": len(top_a & top_b)}
    shap_info = {"selection_rule": "best single model by mean CV ROC-AUC among models "
                                   "with an exact SHAP explainer",
                 "selected_model": selected,
                 "top5_selected": imp[selected].nlargest(5).index.tolist(),
                 "top3_by_model": {m: imp[m].nlargest(3).index.tolist() for m in AGREEMENT_MODELS},
                 "agreement": agree}
    (outdir / "shap_summary.json").write_text(json.dumps(shap_info, indent=2))

    plt.figure()
    shap.plots.beeswarm(explained[selected], show=False, max_display=12)
    plt.title(f"SHAP — {selected}, hold-out ({cfg['short']})", fontsize=9)
    plt.tight_layout(); plt.savefig(figdir / "shap_selected_beeswarm.png", dpi=200); plt.close()

    ranks = imp[AGREEMENT_MODELS].rank(ascending=False).sort_values("LogisticRegression")
    fig, ax = plt.subplots(figsize=(5, 0.35 * len(feats) + 1))
    for m, mk in zip(AGREEMENT_MODELS, ["o", "s", "^"]):
        ax.plot(ranks[m], range(len(ranks)), mk, label=m)
    ax.set_yticks(range(len(ranks))); ax.set_yticklabels(ranks.index, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("SHAP importance rank (1 = most important)")
    ax.legend(fontsize=7); ax.set_title(f"Cross-model SHAP agreement — {cfg['short']}", fontsize=9)
    plt.tight_layout(); plt.savefig(figdir / "shap_agreement.png", dpi=200); plt.close()

    (outdir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"info": info, "statistics": sig, "shap": shap_info}, indent=2))


# -------------------------------------------------------------- compare ---
def compare_stage():
    lines = ["# Cross-dataset summary\n"]
    for key in DATASETS:
        d = ROOT / "results" / key
        if not (d / "statistics.json").exists():
            continue
        info = json.loads((d / "dataset_info.json").read_text())
        sig = json.loads((d / "statistics.json").read_text())
        shp = json.loads((d / "shap_summary.json").read_text())
        cv = pd.read_csv(d / "cv_summary.csv", index_col=0)
        main = next(iter(sig["comparisons"].values()))
        lines += [f"\n## {info['dataset']}",
                  f"- n={info['n_records']} ({info['n_groups']} distinct records), "
                  f"{info['repeats']}x{info['outer_folds']} nested CV",
                  f"- Best single: {sig['best_single']} "
                  f"({cv.loc[sig['best_single'], 'roc_auc_mean']:.3f} ± {cv.loc[sig['best_single'], 'roc_auc_std']:.3f}); "
                  f"best ensemble: {sig['best_ensemble']} "
                  f"({cv.loc[sig['best_ensemble'], 'roc_auc_mean']:.3f} ± {cv.loc[sig['best_ensemble'], 'roc_auc_std']:.3f})",
                  f"- Ensemble − single: {main['mean_diff']:+.4f}, corrected p={main['corrected_p']:.3f}, "
                  f"P(equivalent, ROPE ±{main['rope']})={main['p_equivalent']:.2f}",
                  f"- SHAP ({shp['selected_model']}): {', '.join(shp['top5_selected'])}",
                  f"- SHAP agreement (Kendall τ): " + ", ".join(
                      f"{k}: {v['kendall_tau']:.2f}" for k, v in shp["agreement"].items())]
    out = ROOT / "results" / "cross_dataset_summary.md"
    out.write_text("\n".join(lines)); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--stage", choices=["cv", "final", "compare"], required=True)
    ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="", help="suffix for the results folder, e.g. _hetero")
    ap.add_argument("--dedup", action="store_true", help="remove exact duplicate records before analysis")
    args = ap.parse_args()
    global DEDUP
    DEDUP = args.dedup
    if args.stage == "compare":
        return compare_stage()
    outdir = ROOT / "results" / (args.dataset + args.tag); outdir.mkdir(parents=True, exist_ok=True)
    figdir = ROOT / "figures" / (args.dataset + args.tag); figdir.mkdir(parents=True, exist_ok=True)
    if args.stage == "cv":
        for r in args.repeats:
            run_repeat(args.dataset, r, outdir)
    else:
        final_stage(args.dataset, outdir, figdir)


if __name__ == "__main__":
    main()
