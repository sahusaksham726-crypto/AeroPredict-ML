import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DAY_MAPPING = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
CATEGORICAL_FEATURES = ["sales_channel", "trip_type", "route", "booking_origin"]

print("Training data load ho raha hai...")
train_df = pd.read_csv("data/train_booking_data.csv")

X = train_df.drop(columns=["booking_complete"]).copy()
y = train_df["booking_complete"]

X["flight_day"] = X["flight_day"].map(DAY_MAPPING)
X_enc = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)

print("Lightweight Random Forest train ho raha hai...")
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_enc, y)

joblib.dump(model, "model/ba_booking_rf_model.pkl", compress=3)
print("Done! Naya lightweight model model/ba_booking_rf_model.pkl mein save ho gaya hai.")