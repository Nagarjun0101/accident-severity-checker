

from pathlib import Path

import joblib
import pandas as pd
import sklearn

MODEL_PATH = Path(__file__).parent / "model" / "roadsafety_model.pkl"

# 1. Load everything that the notebook saved (Step 10)
_bundle = joblib.load(MODEL_PATH)

MODEL = _bundle["model"]                      # trained Random Forest
MODEL_NAME = _bundle["model_name"]
COLUMNS = _bundle["columns"]                  # input columns, in training order
WEATHER_OPTIONS = _bundle["weather_options"]  # Clear, Cloudy, Rain, Fog, Snow, Other
MEDIUM_CUTOFF, HIGH_CUTOFF = _bundle["risk_cutoffs"]
CLASSES = [int(c) for c in MODEL.classes_]    # [1, 2, 3, 4]

# dashboard numbers (JSON keys must be text)
STATS = {name: {str(key): float(value) for key, value in table.items()}
         for name, table in _bundle["stats"].items()}

# 2. Warn if scikit-learn differs from the version used for training
if sklearn.__version__ != _bundle["sklearn_version"]:
    print("WARNING: model trained with scikit-learn", _bundle["sklearn_version"],
          "but running", sklearn.__version__)

NOTE = ("Shows how likely an accident under these conditions is to be severe (Severity 3 or 4). "
        "The risk score is relative, not a real-world probability. "
        "Use it to prioritise investigations, not to claim that accidents can be prevented.")


def predict_severity(weather, lighting, hour, weekday, month,
                     visibility, temperature, wind_speed,
                     junction, traffic_signal, crossing):
    """Take the user's inputs and return the model's answer as a dictionary."""
    if weather not in WEATHER_OPTIONS:
        raise ValueError("Unknown weather: " + str(weather))
    if lighting not in ("Day", "Night"):
        raise ValueError("Lighting must be Day or Night")

    # 3. Build one row with exactly the columns the model was trained on
    row = {column: 0 for column in COLUMNS}
    row.update({
        "Visibility": visibility, "Temperature": temperature, "Wind_Speed": wind_speed,
        "Hour": hour, "Weekday": weekday, "Month": month,
        "Is_Night": 1 if lighting == "Night" else 0,
        "Junction": int(junction), "Traffic_Signal": int(traffic_signal), "Crossing": int(crossing),
    })
    weather_column = "Weather_" + weather        # one-hot: Weather_Fog = 1, the rest stay 0
    if weather_column in row:
        row[weather_column] = 1
    X = pd.DataFrame([row], columns=COLUMNS)

    # 4. Predict
    probabilities = MODEL.predict_proba(X)[0]
    scores = {str(c): round(float(p), 4) for c, p in zip(CLASSES, probabilities)}
    risk_score = float(probabilities[CLASSES.index(3)] + probabilities[CLASSES.index(4)])

    if risk_score >= HIGH_CUTOFF:
        level = "High"
    elif risk_score >= MEDIUM_CUTOFF:
        level = "Medium"
    else:
        level = "Low"

    return {
        "predicted_severity": CLASSES[int(probabilities.argmax())],
        "risk_score": round(risk_score, 3),
        "risk_level": level,
        "class_scores": scores,
        "cutoffs": {"medium": round(MEDIUM_CUTOFF, 3), "high": round(HIGH_CUTOFF, 3)},
        "model_name": MODEL_NAME,
        "note": NOTE,
    }
