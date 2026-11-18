import logging

import joblib
import pandas as pd
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("superkart")

superkart_api = Flask("SuperKart")

# Load the serialized pipeline (encoder + tuned Random Forest) once at start-up
model = joblib.load("superkart_model.joblib")
log.info("Model loaded")

NUMERIC = {  # field: (min, max)
    "Product_Weight": (0.01, None),
    "Product_Allocated_Area": (0.0, 1.0),
    "Product_MRP": (0.01, None),
    "Store_Age_Years": (0, 100),
}
CATEGORICAL = {
    "Product_Sugar_Content": {"Low Sugar", "Regular", "No Sugar"},
    "Store_Size": {"Small", "Medium", "High"},
    "Store_Location_City_Type": {"Tier 1", "Tier 2", "Tier 3"},
    "Store_Type": {"Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"},
    "Product_Id_char": {"FD", "DR", "NC"},
    "Product_Type_Category": {"Perishables", "Non Perishables"},
}
FIELDS = list(NUMERIC) + list(CATEGORICAL)


class BadRequest(Exception):
    pass


def validate_frame(df):
    """Raise BadRequest with a clear message if the data is unusable."""
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        raise BadRequest(f"Missing required field(s): {', '.join(missing)}")
    df = df[FIELDS].copy()
    for col, (lo, hi) in NUMERIC.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise BadRequest(f"{col} must be a number in every row")
        if (df[col] < lo).any() or (hi is not None and (df[col] > hi).any()):
            raise BadRequest(f"{col} is out of range (min {lo}" + (f", max {hi})" if hi is not None else ")"))
    for col, allowed in CATEGORICAL.items():
        bad = set(df[col].astype(str)) - allowed
        if bad:
            raise BadRequest(f"{col} has invalid value(s) {sorted(bad)}; allowed: {sorted(allowed)}")
    return df


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object")
    df = validate_frame(pd.DataFrame([data]))
    prediction = float(model.predict(df)[0])
    log.info("Single prediction served")
    return jsonify({"Sales": prediction})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    file = request.files.get("file")
    if file is None or not file.filename:
        raise BadRequest("Upload a CSV file in the 'file' form field")
    if not file.filename.lower().endswith(".csv"):
        raise BadRequest("Only .csv files are accepted")
    try:
        df = pd.read_csv(file)
    except Exception:
        raise BadRequest("The file could not be read as CSV")
    if df.empty:
        raise BadRequest("The CSV file has no rows")
    df = validate_frame(df)
    predictions = [float(p) for p in model.predict(df)]
    log.info("Batch prediction served for %d rows", len(predictions))
    return {str(i): round(p, 2) for i, p in enumerate(predictions)}


@superkart_api.errorhandler(BadRequest)
def handle_bad_request(err):
    log.warning("400: %s", err)
    return jsonify({"error": str(err)}), 400


@superkart_api.errorhandler(Exception)
def handle_unexpected(err):
    log.exception("Unexpected error")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    superkart_api.run(debug=False)
