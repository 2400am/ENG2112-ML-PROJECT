"""Step 4: train and evaluate the next-day WHO exceedance forecast.

Train on 2016-2022, choose each model's alert threshold on 2023 (val),
score ONCE on 2024 (test). Models are compared with two simple baselines.

Reads   data/processed/forecast_table.pkl
Writes  outputs/forecast_results.csv            metrics for every model on val and test
        outputs/forecast_prauc_gain.csv         GBM vs best baseline, with 95% CI
        outputs/forecast_feature_importance.csv which features the GBM relies on
        data/processed/forecast_preds.pkl       every prediction, for plots
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config as c

# load
table = pd.read_pickle(c.PROCESSED / "forecast_table.pkl")
table["y"] = table["target"].astype(int)
table["local_spike"] = table["local_spike"].astype(bool)
table["fire_period"] = table["fire_period"].astype(bool)
table["target_date"] = table["date"] + pd.Timedelta(days=1)

train = table[table["split"] == "train"]
val = table[table["split"] == "val"].copy()
test = table[table["split"] == "test"].copy()
print(f"train {len(train):,} rows ({train['y'].mean():.1%} exceed), "
      f"val {len(val):,} ({val['y'].mean():.1%}), test {len(test):,} ({test['y'].mean():.1%})")

# baselines
# Persistence: "tomorrow will be like today" - today's PM2.5 so far is the score.
for df in (val, test):
    df["persistence"] = df["pm_today"]

# climatology: how often this station exceeded in this calendar month during training.
month = lambda df: df["target_date"].dt.month
clim = train.assign(month=month(train)).groupby(["site_id", "month"])["y"].mean().rename("climatology")
for df in (val, test):
    df["climatology"] = df.assign(month=month(df)).join(clim, on=["site_id", "month"])["climatology"].values

# models
X_train, y_train = train[c.FEATURES], train["y"]

# logistic regression: a straight line model. It can't handle missing values or very
# different scales, so missing values are filled with the median and features are scaled.
logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)

# gradient boosting: hundreds of small decision trees, each correcting the last.
# handles missing values and non-linear effects (e.g. "calm AND cold AND high today").
gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                     min_samples_leaf=50, l2_regularization=1.0,
                                     random_state=c.SEED)
gbm.fit(X_train, y_train)

for df in (val, test):
    df["logreg"] = logreg.predict_proba(df[c.FEATURES])[:, 1]
    df["gbm"] = gbm.predict_proba(df[c.FEATURES])[:, 1]

MODELS = ["persistence", "climatology", "logreg", "gbm"]

# alert thresholds, chosen on val only
def best_threshold(y, score):
    """Threshold on `score` that gives the best F1 (balance of catching and false alarms)."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return thresholds[np.argmax(f1[:-1])]

thresholds = {m: best_threshold(val["y"], val[m].fillna(0)) for m in MODELS}

# metrics
def metrics(y, score, thr):
    score = score.fillna(0)
    alert = score >= thr
    return {
        "n": len(y), "exceed_days": int(y.sum()),
        "PR_AUC": average_precision_score(y, score),
        "ROC_AUC": roc_auc_score(y, score),
        "recall": recall_score(y, alert, zero_division=0),
        "precision": precision_score(y, alert, zero_division=0),
        "F1": f1_score(y, alert, zero_division=0),
        "Brier": brier_score_loss(y, score) if score.between(0, 1).all() else np.nan,
        "threshold": thr,
    }

rows = []
for split_name, df in [("val", val), ("test", test)]:
    strata = {"all": df, "excl_local_spikes": df[~df["local_spike"]]}
    for stratum, d in strata.items():
        if d["y"].nunique() < 2:
            continue
        for m in MODELS:
            rows.append({"split": split_name, "stratum": stratum, "model": m,
                         **metrics(d["y"], d[m], thresholds[m])})
results = pd.DataFrame(rows)
results.to_csv(c.OUTPUTS / "forecast_results.csv", index=False)

# is GBM really better? 95% confidence interval
# resample whole calendar weeks (days within a week are correlated), recompute the
# PR-AUC gain each time, and take the middle 95% of the results.
baselines = ["persistence", "climatology"]
best_base = max(baselines, key=lambda m: average_precision_score(val["y"], val[m].fillna(0)))

def prauc_gain(d):
    return average_precision_score(d["y"], d["gbm"]) - average_precision_score(d["y"], d[best_base].fillna(0))

week = test["target_date"].dt.to_period("W")
groups = {w: g for w, g in test.groupby(week)}
week_ids = list(groups)
rng = np.random.default_rng(c.SEED)
boot = []
for _ in range(c.N_BOOT):
    sample = pd.concat([groups[w] for w in rng.choice(week_ids, len(week_ids))])
    if sample["y"].nunique() == 2:
        boot.append(prauc_gain(sample))
gain = prauc_gain(test)
lo, hi = np.percentile(boot, [2.5, 97.5])
meets = gain >= c.MIN_PRAUC_GAIN and lo > 0
pd.DataFrame([{"comparison": f"gbm vs {best_base}", "PR_AUC_gain": gain, "CI_low": lo, "CI_high": hi,
               "meets_criterion": meets}]).to_csv(c.OUTPUTS / "forecast_prauc_gain.csv", index=False)

# which features matter
# Shuffle one feature at a time on the test set and see how much PR-AUC drops.
imp = permutation_importance(gbm, test[c.FEATURES], test["y"], scoring="average_precision",
                             n_repeats=5, random_state=c.SEED)
importance = pd.DataFrame({"feature": c.FEATURES, "PR_AUC_drop": imp.importances_mean,
                           "std": imp.importances_std}).sort_values("PR_AUC_drop", ascending=False)
importance.to_csv(c.OUTPUTS / "forecast_feature_importance.csv", index=False)

# save
keep = ["site_id", "date", "target_date", "target_pm", "y", "split", "fire_period", "local_spike"] + MODELS
pd.concat([val, test])[keep].to_pickle(c.PROCESSED / "forecast_preds.pkl")

# for report
pd.set_option("display.width", 200)
show = ["model", "exceed_days", "PR_AUC", "ROC_AUC", "recall", "precision", "F1", "Brier"]
for (split_name, stratum), r in results.groupby(["split", "stratum"], sort=False):
    print(f"\n{split_name.upper()} - {stratum}")
    print(r[show].round(3).to_string(index=False))
print(f"\nPR-AUC gain, GBM vs {best_base} (test): {gain:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
      f"-> meets pre-registered criterion (gain >= {c.MIN_PRAUC_GAIN}, CI above 0): {meets}")
print("\nTop features (PR-AUC drop when shuffled):")
print(importance.head(8).round(4).to_string(index=False))