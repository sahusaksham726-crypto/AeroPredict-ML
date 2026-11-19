import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ba_booking_rf_model.pkl")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train_booking_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test_booking_data.csv")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-for-deployment")

DAY_MAPPING = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
NUMERIC_FEATURES = [
    "num_passengers", "purchase_lead", "length_of_stay", "flight_hour",
    "flight_day", "wants_extra_baggage", "wants_preferred_seat",
    "wants_in_flight_meals", "flight_duration"
]
CATEGORICAL_FEATURES = ["sales_channel", "trip_type", "route", "booking_origin"]

model = None
train_df = None
test_df = None
load_error = None
cached_stats = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file missing. Put ba_booking_rf_model.pkl inside the model/ folder.")
    model = joblib.load(MODEL_PATH)
    if os.path.exists(TRAIN_PATH):
        train_df = pd.read_csv(TRAIN_PATH)
    if os.path.exists(TEST_PATH):
        test_df = pd.read_csv(TEST_PATH)
except Exception as exc:
    load_error = str(exc)

def model_columns():
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise RuntimeError("The saved model does not expose its expected feature names.")

def prepare_one_row(raw):
    """Recreate the notebook's day mapping and get_dummies(drop_first=True) encoding."""
    row = pd.DataFrame([raw])
    row["flight_day"] = row["flight_day"].map(DAY_MAPPING)
    encoded = pd.get_dummies(
        row,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
        dtype=int
    )
    if train_df is None:
        raise RuntimeError("Training CSV is required to reproduce categorical encoding.")
    for col in CATEGORICAL_FEATURES:
        baseline = sorted(train_df[col].dropna().astype(str).unique().tolist())[0]
        baseline_dummy = f"{col}_{baseline}"
        if baseline_dummy in encoded.columns:
            encoded = encoded.drop(columns=[baseline_dummy])
    expected = model_columns()
    encoded = encoded.reindex(columns=expected, fill_value=0)
    return encoded

def category_options(column):
    if train_df is None:
        return []
    return sorted(train_df[column].dropna().astype(str).unique().tolist())

def get_dashboard_stats():
    global cached_stats
    if cached_stats is not None:
        return cached_stats

    stats = {"total_train": 0, "total_test": 0, "completion_rate": None, "accuracy": None,
             "precision": None, "recall": None, "f1": None, "class0": None, "class1": None}
    if train_df is not None:
        stats["total_train"] = len(train_df)
        if "booking_complete" in train_df:
            stats["completion_rate"] = round(float(train_df["booking_complete"].mean()) * 100, 2)
            counts = train_df["booking_complete"].value_counts().to_dict()
            stats["class0"] = int(counts.get(0, 0))
            stats["class1"] = int(counts.get(1, 0))
    if test_df is not None:
        stats["total_test"] = len(test_df)
        if model is not None and "booking_complete" in test_df:
            X = test_df.drop(columns=["booking_complete"]).copy()
            X["flight_day"] = X["flight_day"].map(DAY_MAPPING)
            X_enc = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True, dtype="int8")
            X_enc = X_enc.reindex(columns=model_columns(), fill_value=0)
            y_true = test_df["booking_complete"]
            y_pred = model.predict(X_enc)
            stats["accuracy"] = round(accuracy_score(y_true, y_pred) * 100, 2)
            stats["precision"] = round(precision_score(y_true, y_pred, zero_division=0) * 100, 2)
            stats["recall"] = round(recall_score(y_true, y_pred, zero_division=0) * 100, 2)
            stats["f1"] = round(f1_score(y_true, y_pred, zero_division=0) * 100, 2)
    cached_stats = stats
    return stats

@app.route("/")
def home():
    stats = get_dashboard_stats()
    return render_template("index.html", stats=stats)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form_data = request.form.to_dict() if request.method == "POST" else {}
    options = {
        "sales_channel": category_options("sales_channel"),
        "trip_type": category_options("trip_type"),
        "route": category_options("route"),
        "booking_origin": category_options("booking_origin"),
    }
    if request.method == "POST":
        try:
            raw = {
                "num_passengers": int(request.form["num_passengers"]),
                "sales_channel": request.form["sales_channel"],
                "trip_type": request.form["trip_type"],
                "purchase_lead": int(request.form["purchase_lead"]),
                "length_of_stay": int(request.form["length_of_stay"]),
                "flight_hour": int(request.form["flight_hour"]),
                "flight_day": request.form["flight_day"],
                "route": request.form["route"],
                "booking_origin": request.form["booking_origin"],
                "wants_extra_baggage": int(request.form["wants_extra_baggage"]),
                "wants_preferred_seat": int(request.form["wants_preferred_seat"]),
                "wants_in_flight_meals": int(request.form["wants_in_flight_meals"]),
                "flight_duration": float(request.form["flight_duration"]),
            }
            if not 0 <= raw["flight_hour"] <= 23:
                raise ValueError("Flight hour must be between 0 and 23.")
            if raw["num_passengers"] < 1 or raw["purchase_lead"] < 0 or raw["length_of_stay"] < 0 or raw["flight_duration"] <= 0:
                raise ValueError("Please enter valid positive values for passengers, lead time, stay, and duration.")
            row = prepare_one_row(raw)
            prediction = int(model.predict(row)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(row)[0][list(model.classes_).index(prediction)]) * 100
            result = {
                "label": "Booking completed" if prediction == 1 else "Booking not completed",
                "prediction": prediction,
                "probability": round(probability, 2) if probability is not None else None,
                "positive": prediction == 1
            }
        except Exception as exc:
            flash(f"Could not generate prediction: {exc}", "error")
    return render_template("predict.html", options=options, result=result, form_data=form_data)

@app.route("/dashboard")
def dashboard():
    stats = get_dashboard_stats()
    chart = None
    if train_df is not None and "booking_complete" in train_df:
        counts = train_df["booking_complete"].value_counts().sort_index()
        chart = {"labels": ["Not completed (0)", "Completed (1)"],
                 "values": [int(counts.get(0, 0)), int(counts.get(1, 0))]}
    return render_template("dashboard.html", stats=stats, chart=chart)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    if load_error:
        print("Startup warning:", load_error)
    app.run(debug=True)