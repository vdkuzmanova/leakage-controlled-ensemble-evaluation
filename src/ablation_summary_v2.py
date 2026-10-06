"""Ablation summary with (i) ROC-AUC alongside accuracy, (ii) evaluation restricted to original (non-synthetic)
test records for protocols that oversample before splitting, and (iii) a like-for-like ensemble comparison:
per split, the better of the two ensembles versus the better of its own base learners (Random Forest, XGBoost)."""
import glob
import pandas as pd
from pipeline import ROOT

ENS = ["Voting(RF+XGB)", "Stacking(RF+XGB->LR)"]; BASE = ["RandomForest", "XGBoost"]
ORDER = ["L0-group", "L0", "L1", "L2", "L3", "L4"]

def load():
    v1 = pd.concat([pd.read_csv(p) for p in glob.glob(str(ROOT / "results/ablation/*_L*.csv"))])
    v2 = pd.concat([pd.read_csv(p) for p in glob.glob(str(ROOT / "results/ablation_v2/*_L*.csv"))])
    v1 = v1[~v1.protocol.isin(["L2", "L3", "L4"])]          # v2 reproduces v1 exactly and adds orig_* columns
    return pd.concat([v1, v2], ignore_index=True)

def lead(g, metric, thr=0.01):
    w = g.pivot_table(index="seed", columns="model", values=metric)
    d = w[ENS].max(axis=1) - w[BASE].max(axis=1)
    return 100 * (d >= thr).mean(), 100 * (d <= -thr).mean()

def main():
    df = load(); rows = []
    for ds in ["pima", "early_stage"]:
        for p in [x for x in ORDER if ((df.dataset == ds) & (df.protocol == x)).any()]:
            g = df[(df.dataset == ds) & (df.protocol == p)]; m = g.groupby("model").mean(numeric_only=True)
            has_orig = "orig_roc_auc" in g and g["orig_roc_auc"].notna().all()
            ea, ba = lead(g, "accuracy"); eu, bu = lead(g, "roc_auc")
            r = {"dataset": ds, "protocol": p, "mean_acc": m.accuracy.mean(), "mean_auc": m.roc_auc.mean(),
                 "rf_acc": m.accuracy["RandomForest"], "lr_acc": m.accuracy["LogisticRegression"],
                 "rf_auc": m.roc_auc["RandomForest"], "lr_auc": m.roc_auc["LogisticRegression"],
                 "best_acc_median": g.groupby("seed").accuracy.max().median(),
                 "best_acc_max": g.groupby("seed").accuracy.max().max(),
                 "best_acc_p90": g.groupby("seed").accuracy.max().quantile(0.9),
                 "ens_lead_acc_pct": ea, "base_lead_acc_pct": ba, "ens_lead_auc_pct": eu, "base_lead_auc_pct": bu}
            if has_orig:
                eo, bo = lead(g, "orig_roc_auc"); eao, bao = lead(g, "orig_accuracy")
                r.update({"orig_mean_acc": m.orig_accuracy.mean(), "orig_mean_auc": m.orig_roc_auc.mean(),
                          "orig_rf_auc": m.orig_roc_auc["RandomForest"], "orig_lr_auc": m.orig_roc_auc["LogisticRegression"],
                          "orig_lr_acc": m.orig_accuracy["LogisticRegression"], "orig_rf_acc": m.orig_accuracy["RandomForest"],
                          "orig_ens_lead_auc_pct": eo, "orig_base_lead_auc_pct": bo,
                          "orig_ens_lead_acc_pct": eao, "orig_base_lead_acc_pct": bao,
                          "n_test": g.n_test.mean(), "n_test_orig": g.n_test_orig.mean(),
                          "test_pos_rate": g.test_pos_rate.mean(), "test_pos_rate_orig": g.test_pos_rate_orig.mean()})
            rows.append(r)
    s = pd.DataFrame(rows); s.to_csv(ROOT / "results/ablation_v2/ablation_summary_v2.csv", index=False)
    pd.set_option("display.width", 250); print(s.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
