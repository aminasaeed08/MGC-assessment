"""Scoring helper used by the web app.  Loads (or trains, on first run) the model and scores one lead."""
import numpy as np
import pandas as pd

import joblib
from ml import train
from ml.features import prepare

_bundle = None


def get_bundle() -> dict:
    """Load ml/model.joblib; if it is missing or unreadable (e.g. different scikit-learn version), train it. Takes ~2 s."""
    global _bundle
    if _bundle is None:
        try:
            _bundle = joblib.load(train.MODEL_PATH)
        except Exception:
            train.main()
            _bundle = joblib.load(train.MODEL_PATH)
    return _bundle


def score_lead(raw: dict) -> dict:
    """raw = the original CRM column names (budget_pkr_lac, city, source, site_visits, ...). Missing -> NaN."""
    b = get_bundle()
    cols = ["source", "city", "property_type", "budget_pkr_lac", "bedrooms", "first_response_minutes", "calls_made",
            "total_call_seconds", "whatsapp_replies", "site_visits", "agent_experience_years",
            "is_overseas", "referred_by_existing_client", "has_financing_approved"]
    row = pd.DataFrame([{c: raw.get(c, np.nan) for c in cols}])
    feats = prepare(row)
    prob = float(b["model"].predict_proba(feats[b["cat"] + b["num"]])[:, 1][0])
    ref = b["ref_scores"]
    top_pct = float((ref >= prob).mean() * 100)      # share of recent (held-out) leads that scored at least as high
    return {"probability": prob, "percent": round(prob * 100, 1), "top_percent_of_recent_leads": round(top_pct, 1)}
