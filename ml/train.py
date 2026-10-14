"""Part 3: lead-conversion scoring.   python -m ml.train

Decisions (full reasoning in README):
  * de-duplicate FIRST (160 exact copies would otherwise straddle train/test)
  * drop token_amount_received_pkr (leak), ids, created_at, area
  * split by TIME: oldest 80% train, newest 20% test
  * baseline = logistic regression (no tuning)
  * metric = ROC-AUC vs 0.5 no-skill, translated to 'conversions in the top 100 called'
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features import AT_CREATION_NUM, CAT_FEATURES, DROPPED, NUM_FEATURES, prepare

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ml" / "model.joblib"
METRICS_PATH = ROOT / "ml" / "metrics.json"
TOP_K = 100
TRAIN_FRACTION = 0.80


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "leads.csv", parse_dates=["created_at"])
    # Same canonical-row rule as queries.sql (earliest created_at, then smallest lead_id).
    df = df.sort_values(["created_at", "lead_id"]).drop_duplicates("crm_record_hash", keep="first")
    return prepare(df).reset_index(drop=True)


def build_pipeline(cat, num) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # + 'was missing' flags
            ("scale", StandardScaler()),
        ]), num),
    ])
    # No class_weight: we only need a ranking, and leaving it unweighted keeps the
    # probabilities roughly calibrated, which the web app displays as a percentage.
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def top_k_conversions(y: np.ndarray, p: np.ndarray, k: int = TOP_K) -> int:
    return int(y[np.argsort(-p, kind="stable")[:k]].sum())


def evaluate(model, cat, num, train, test) -> dict:
    model.fit(train[cat + num], train["converted"])
    p = model.predict_proba(test[cat + num])[:, 1]
    y = test["converted"].to_numpy()
    return {"auc": roc_auc_score(y, p), "top_k": top_k_conversions(y, p), "p": p}


def main():
    df = load_clean()
    n_train = int(len(df) * TRAIN_FRACTION)
    train, test = df.iloc[:n_train], df.iloc[n_train:]          # df is already sorted by created_at
    assert train["created_at"].max() <= test["created_at"].min()  # no time overlap

    base_rate = float(test["converted"].mean())
    cat, num = CAT_FEATURES, NUM_FEATURES

    main_res = evaluate(build_pipeline(cat, num), cat, num, train, test)
    y, p = test["converted"].to_numpy(), main_res["p"]

    # Uncertainty on the headline number: bootstrap the test set.
    rng = np.random.default_rng(0)
    boot = [top_k_conversions(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]

    # Ablation 1: only what is known when the lead lands (no calls / WhatsApp / visits).
    at_creation = evaluate(build_pipeline(cat, AT_CREATION_NUM), cat, AT_CREATION_NUM, train, test)

    # Ablation 2: what happens if we (wrongly) keep the leaking column.
    leak = df.copy()
    leak_num = num + ["token_amount_received_pkr"]
    leak_res = evaluate(build_pipeline(cat, leak_num), cat, leak_num, leak.iloc[:n_train], leak.iloc[n_train:])

    final_model = build_pipeline(cat, num).fit(train[cat + num], train["converted"])  # the one the app serves

    metrics = {
        "rows_after_dedup": len(df),
        "train_rows": len(train), "test_rows": len(test),
        "train_period": [str(train.created_at.min()), str(train.created_at.max())],
        "test_period": [str(test.created_at.min()), str(test.created_at.max())],
        "train_base_rate": float(train["converted"].mean()),
        "test_base_rate": base_rate,
        "test_conversions": int(y.sum()),
        "model_auc": main_res["auc"], "no_skill_auc": 0.5,
        "top_k": TOP_K,
        "model_top_k_conversions": main_res["top_k"],
        "model_top_k_ci95": [int(np.percentile(boot, 2.5)), int(np.percentile(boot, 97.5))],
        "no_skill_top_k_conversions": TOP_K * base_rate,
        "lift_vs_random": main_res["top_k"] / (TOP_K * base_rate),
        "share_of_all_test_conversions_in_top_k": main_res["top_k"] / int(y.sum()),
        "at_creation_only": {"auc": at_creation["auc"], "top_k_conversions": at_creation["top_k"]},
        "WITH_LEAK_token_amount": {"auc": leak_res["auc"], "top_k_conversions": leak_res["top_k"]},
        "dropped_columns": DROPPED,
    }
    joblib.dump({"model": final_model, "cat": cat, "num": num, "ref_scores": np.sort(p),
                 "choices": {c: sorted(df[c].dropna().unique().tolist()) for c in cat}}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def print_report(m: dict):
    k = m["top_k"]
    print(f"Leads after de-dup: {m['rows_after_dedup']}  |  train {m['train_rows']} ({m['train_period'][0][:10]} -> {m['train_period'][1][:10]})"
          f"  |  test {m['test_rows']} ({m['test_period'][0][:10]} -> {m['test_period'][1][:10]})")
    print(f"Test base conversion rate: {m['test_base_rate']:.1%}  ({m['test_conversions']} conversions)")
    print(f"ROC-AUC            model {m['model_auc']:.3f}   no-skill 0.500")
    print(f"Top-{k} called      model {m['model_top_k_conversions']} conversions (95% CI {m['model_top_k_ci95'][0]}-{m['model_top_k_ci95'][1]})"
          f"   no-skill ~{m['no_skill_top_k_conversions']:.1f}   => {m['lift_vs_random']:.1f}x lift")
    print(f"                   top-{k} captures {m['share_of_all_test_conversions_in_top_k']:.0%} of ALL conversions in the test period")
    print(f"At-creation-only   AUC {m['at_creation_only']['auc']:.3f}, top-{k} = {m['at_creation_only']['top_k_conversions']}")
    print(f"WITH token leak    AUC {m['WITH_LEAK_token_amount']['auc']:.3f}, top-{k} = {m['WITH_LEAK_token_amount']['top_k_conversions']}   <- do not trust; this is why it is dropped")


if __name__ == "__main__":
    print_report(main())
