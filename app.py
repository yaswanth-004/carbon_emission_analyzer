"""
Carbon Emission Prediction — Flask Application
Loads Decision Tree Regressor + Classifier from .pkl files
and serves a prediction UI.
"""

import os
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ─────────────────────────────────────────────
# Load Models & Metadata  (once at startup)
# ─────────────────────────────────────────────
#print("Loading models...")
reg_model   = joblib.load(os.path.join(MODEL_DIR, "model_regression.pkl"))
cls_model   = joblib.load(os.path.join(MODEL_DIR, "model_classification.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
label_maps  = joblib.load(os.path.join(MODEL_DIR, "label_maps.pkl"))
#print(f"✅ Models loaded — {len(feature_names)} features")

# ─────────────────────────────────────────────
# Label maps  (label → encoded int)
# ─────────────────────────────────────────────
BODY_TYPE   = label_maps.get("body_type",   {})
SEX         = label_maps.get("sex",         {})
DIET        = label_maps.get("diet",        {})
SHOWER      = label_maps.get("how_often_shower",        {})
HEATING     = label_maps.get("heating_energy_source",   {})
TRANSPORT   = label_maps.get("transport",   {})
VEHICLE     = label_maps.get("vehicle_type",{})
SOCIAL      = label_maps.get("social_activity",         {})
AIR_TRAVEL  = label_maps.get("frequency_of_traveling_by_air", {})
BAG_SIZE    = label_maps.get("waste_bag_size", {})
EFFICIENCY  = label_maps.get("energy_efficiency", {})
RECYCLING   = label_maps.get("recycling",   {})
COOKING     = label_maps.get("cooking_with",{})
TYPE_MAP    = label_maps.get("type",        {"H": 0, "I": 1})

BAND_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
BAND_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
BAND_ICONS  = {0: "🟢", 1: "🟡", 2: "🔴"}

# ─────────────────────────────────────────────
# Helper: Build input DataFrame from form data
# ─────────────────────────────────────────────
def encode_household(form):
    """Encode household form fields into a numeric row."""

    def enc(mapping, key, default="0"):
        val = form.get(key, default).strip().lower()
        # Try exact match first, then fuzzy
        if val in mapping:
            return mapping[val]
        for k, v in mapping.items():
            if k.lower() == val:
                return v
        return 0

    # recycling & cooking: multi-select sorted join
    recycling_raw = sorted(form.getlist("recycling"))
    cooking_raw   = sorted(form.getlist("cooking_with"))
    recycling_str = "_".join(recycling_raw) if recycling_raw else ""
    cooking_str   = "_".join(cooking_raw)   if cooking_raw   else ""

    return {
        "body_type":                      enc(BODY_TYPE,  "body_type"),
        "sex":                            enc(SEX,        "sex"),
        "diet":                           enc(DIET,       "diet"),
        "how_often_shower":               enc(SHOWER,     "how_often_shower"),
        "heating_energy_source":          enc(HEATING,    "heating_energy_source"),
        "transport":                      enc(TRANSPORT,  "transport"),
        "vehicle_type":                   enc(VEHICLE,    "vehicle_type"),
        "social_activity":                enc(SOCIAL,     "social_activity"),
        "monthly_grocery_bill":           float(form.get("monthly_grocery_bill", 200)),
        "frequency_of_traveling_by_air":  enc(AIR_TRAVEL, "frequency_of_traveling_by_air"),
        "vehicle_monthly_distance_km":    float(form.get("vehicle_monthly_distance_km", 0)),
        "waste_bag_size":                 enc(BAG_SIZE,   "waste_bag_size"),
        "waste_bag_weekly_count":         float(form.get("waste_bag_weekly_count", 2)),
        "how_long_tv_pc_daily_hour":      float(form.get("how_long_tv_pc_daily_hour", 3)),
        "how_many_new_clothes_monthly":   float(form.get("how_many_new_clothes_monthly", 5)),
        "how_long_internet_daily_hour":   float(form.get("how_long_internet_daily_hour", 3)),
        "energy_efficiency":              enc(EFFICIENCY, "energy_efficiency"),
        "recycling":                      RECYCLING.get(recycling_str, 0),
        "cooking_with":                   COOKING.get(cooking_str, 0),
        # Industrial columns zeroed for household
        "type":    TYPE_MAP.get("H", 0),
        "country": 0,
        "year":    0,
        "coal_co2":            0,
        "oil_co2":             0,
        "gas_co2":             0,
        "cement_co2":          0,
        "flaring_co2":         0,
        "other_industry_co2":  0,
    }


def encode_industrial(form):
    """Encode industrial form fields into a numeric row."""
    return {
        # Household columns zeroed for industrial
        "body_type": 0, "sex": 0, "diet": 0, "how_often_shower": 0,
        "heating_energy_source": 0, "transport": 0, "vehicle_type": 0,
        "social_activity": 0, "monthly_grocery_bill": 0,
        "frequency_of_traveling_by_air": 0,
        "vehicle_monthly_distance_km": 0, "waste_bag_size": 0,
        "waste_bag_weekly_count": 0, "how_long_tv_pc_daily_hour": 0,
        "how_many_new_clothes_monthly": 0, "how_long_internet_daily_hour": 0,
        "energy_efficiency": 0, "recycling": 0, "cooking_with": 0,
        # Industrial fields
        "type":               TYPE_MAP.get("I", 1),
        "country":            float(form.get("country_code", 0)),
        "year":               float(form.get("year", 2020)),
        "coal_co2":           float(form.get("coal_co2",           0)) * 1e9,
        "oil_co2":            float(form.get("oil_co2",            0)) * 1e9,
        "gas_co2":            float(form.get("gas_co2",            0)) * 1e9,
        "cement_co2":         float(form.get("cement_co2",         0)) * 1e9,
        "flaring_co2":        float(form.get("flaring_co2",        0)) * 1e9,
        "other_industry_co2": float(form.get("other_industry_co2", 0)) * 1e9,
    }


def run_prediction(row_dict):
    """Given a dict of encoded values, return prediction results."""
    df = pd.DataFrame([row_dict])[feature_names]
    co2_kg    = float(reg_model.predict(df)[0])
    band_code = int(cls_model.predict(df)[0])
    band      = BAND_LABELS.get(band_code, "UNKNOWN")
    color     = BAND_COLORS.get(band_code, "#888")
    icon      = BAND_ICONS.get(band_code, "⚪")
    co2_tonnes = co2_kg / 1000
    return {
        "co2_kg":     round(co2_kg, 2),
        "co2_tonnes": round(co2_tonnes, 4),
        "band":       band,
        "band_code":  band_code,
        "color":      color,
        "icon":       icon,
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    context = {
        "body_types":   [k for k in BODY_TYPE  if k != "0"],
        "sexes":        [k for k in SEX         if k != "0"],
        "diets":        [k for k in DIET        if k != "0"],
        "showers":      [k for k in SHOWER      if k != "0"],
        "heatings":     [k for k in HEATING     if k != "0"],
        "transports":   [k for k in TRANSPORT   if k != "0"],
        "vehicles":     [k for k in VEHICLE     if k != "0"],
        "socials":      [k for k in SOCIAL      if k != "0"],
        "air_travels":  [k for k in AIR_TRAVEL  if k != "0"],
        "bag_sizes":    [k for k in BAG_SIZE     if k != "0"],
        "efficiencies": [k for k in EFFICIENCY  if k != "0"],
        "recycling_opts": [k for k in RECYCLING if k not in ("", "0")],
        "cooking_opts":   [k for k in COOKING   if k not in ("", "0")],
    }
    return render_template("index.html", **context)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        mode = request.form.get("mode", "household")
        if mode == "household":
            row = encode_household(request.form)
        else:
            row = encode_industrial(request.form)
        result = run_prediction(row)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "features": len(feature_names),
        "reg_model": str(type(reg_model).__name__),
        "cls_model": str(type(cls_model).__name__),
    })


# ─────────────────────────────────────────────
if __name__ == "__main__":
   port = int(os.environ.get("PORT", 5000))
   app.run(host="0.0.0.0", port=port)
