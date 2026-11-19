# British Airways Customer Booking Prediction — Flask Web App

A Flask demonstration website for the supplied Random Forest model predicting `booking_complete`.

## Project structure
- `app.py`: Flask routes, feature preparation, predictions and metrics
- `templates/`: HTML pages
- `static/css/style.css`: responsive gradient styling
- `data/`: place the supplied training and test CSV files here
- `model/`: place `ba_booking_rf_model.pkl` here

## Setup (recommended: Python 3.10 or 3.11)
Create and activate a virtual environment:

### Windows
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy these files into the project:
- `train_booking_data.csv` → `data/train_booking_data.csv`
- `test_booking_data.csv` → `data/test_booking_data.csv`
- `ba_booking_rf_model.pkl` → `model/ba_booking_rf_model.pkl`

Run:
```bash
python app.py
```
Open `http://127.0.0.1:5000`.

## Compatibility note
The model file was saved using scikit-learn 1.5.1. The requirements pin that version to reduce pickle compatibility issues. Only load pickle/joblib files from trusted sources.

## Important modelling note
The training notebook uses `pd.get_dummies(..., drop_first=True)` and aligns the test columns to the training feature set. The web app follows the same encoding approach and aligns input columns to `model.feature_names_in_`.

## Metrics
Dashboard metrics are computed from the supplied test CSV if available. Accuracy is shown alongside precision, recall and F1-score for the positive class (`booking_complete = 1`).

## Disclaimer
This is an academic project demo. Predictions are not guaranteed and should not be used for real-world airline operations without further validation.
