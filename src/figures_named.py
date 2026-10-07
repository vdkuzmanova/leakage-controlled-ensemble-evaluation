"""Re-draws the hold-out ROC and SHAP figures with the model names used in the manuscript tables."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, shap, sys
from sklearn.metrics import roc_curve, roc_auc_score
from pipeline import load, holdout_split, tune_all, shap_values, ROOT, RANDOM_STATE
NAMES = {"LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree", "RandomForest": "Random Forest",
         "SVM": "SVM", "XGBoost": "XGBoost", "NeuralNetwork": "Multi-Layer Perceptron (MLP)", "Voting(RF+XGB)": "Voting (RF+XGB)",
         "Stacking(RF+XGB->LR)": "Stacking (RF+XGB→LR)", "Voting(all)": "Voting (five-model)",
         "Stacking(all->LR)": "Stacking (five-model→LR)"}
STYLE = ["-", "--", "-.", ":"]
for key, short in [(sys.argv[1], sys.argv[2])]:
    cfg, X, y, groups = load(key); feats = cfg["numeric"] + cfg["categorical"]
    tr, te = holdout_split(y, groups)
    fitted, _ = tune_all(X.iloc[tr], y.iloc[tr], groups[tr], cfg["numeric"], cfg["categorical"], RANDOM_STATE)
    order = pd.read_csv(ROOT / "results" / (key + "_hetero") / "cv_summary.csv", index_col=0).index
    fig, ax = plt.subplots(figsize=(6.6, 5.6)); cmap = plt.get_cmap("tab10")
    for i, m in enumerate(order):
        prob = fitted[m].predict_proba(X.iloc[te])[:, 1]; fpr, tpr, _ = roc_curve(y.iloc[te], prob)
        ax.plot(fpr, tpr, color=cmap(i % 10), ls=STYLE[i % 4], lw=1.4,
                label="%s (AUC = %.3f)" % (NAMES[m], roc_auc_score(y.iloc[te], prob)))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Hold-out ROC curves — %s" % short, fontsize=10); ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(ROOT / "figures" / (key + "_hetero") / "roc_holdout_named.png", dpi=300); plt.savefig(ROOT / "figures" / (key + "_hetero") / "roc_holdout_named.pdf"); plt.close()
    e = shap_values("RandomForest", fitted["RandomForest"], X.iloc[tr], X.iloc[te], feats)
    plt.figure(); shap.plots.beeswarm(e, show=False, max_display=12)
    plt.title("SHAP values — Random Forest, hold-out set (%s)" % short, fontsize=9)
    plt.tight_layout(); plt.savefig(ROOT / "figures" / (key + "_hetero") / "shap_rf_named.png", dpi=300); plt.savefig(ROOT / "figures" / (key + "_hetero") / "shap_rf_named.pdf"); plt.close()
    print(key, "done")
