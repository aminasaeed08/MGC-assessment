"""Shared feature preparation: used by BOTH training and the web app, so a lead is
scored exactly the way the model was trained (no train/serve skew)."""
import numpy as np
import pandas as pd

CITY_ALIASES = {"ISB": "Islamabad", "RWP": "Rawalpindi", "KHI": "Karachi"}

# Columns the model is allowed to see ------------------------------------------
CAT_FEATURES = ["source", "city", "property_type"]
NUM_FEATURES = [
    "budget_log", "bedrooms", "first_response_log", "calls_made", "call_seconds_log",
    "whatsapp_replies", "site_visits", "agent_experience_years",
    "is_overseas", "referred_by_existing_client", "has_financing_approved",
]
# Subset known at the moment the lead arrives (no calls/WhatsApp/visits yet) - used for the ablation.
AT_CREATION_NUM = [c for c in NUM_FEATURES
                   if c not in ("calls_made", "call_seconds_log", "whatsapp_replies", "site_visits")]

# Columns deliberately NOT used (reason in README) ------------------------------
DROPPED = {
    "token_amount_received_pkr": "TARGET LEAK: paid at/after conversion; > 0 for 100% of converters",
    "lead_id": "identifier (row number, not behaviour)",
    "crm_record_hash": "identifier / dedup fingerprint",
    "created_at": "used only to order the time split; raw timestamp would not generalise",
    "area": "10 near-identical categories, conversion differences are within noise",
}


def normalise_city(value) -> str:
    """Same rules as Query 3 in queries.sql."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().upper()
    return CITY_ALIASES.get(s, s.title())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw lead columns -> model features. Missing values stay NaN (the pipeline imputes
    them and adds a 'was missing' flag, so 'bedrooms missing' (plots/shops) is itself a signal)."""
    out = df.copy()
    out["city"] = out["city"].map(normalise_city)
    out["budget_log"] = np.log1p(out["budget_pkr_lac"])                # heavy right tail
    out["first_response_log"] = np.log1p(out["first_response_minutes"])  # max 3,107 min
    out["call_seconds_log"] = np.log1p(out["total_call_seconds"])
    return out
