from pathlib import Path
import pickle
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_model.pkl"

app = Flask(__name__)
model = None

def load_model():
    global model
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
    else:
        model = None

load_model()

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

OPTIONS = {
    "sex": [("1", "Male (1)"), ("0", "Female (0)")],
    "cp": [("1", "Typical angina (1)"), ("2", "Atypical angina (2)"),
           ("3", "Non-anginal pain (3)"), ("4", "Asymptomatic (4)")],
    "fbs": [("1", "Yes (1)"), ("0", "No (0)")],
    "restecg": [("0", "Normal (0)"), ("1", "ST-T wave abnormality (1)"),
                ("2", "Left ventricular hypertrophy (2)")],
    "exang": [("1", "Yes (1)"), ("0", "No (0)")],
    "slope": [("1", "Upsloping (1)"), ("2", "Flat (2)"), ("3", "Downsloping (3)")],
    "ca": [(str(i), f"{i} major vessels (0–3)") for i in range(4)],
    "thal": [("3", "Normal (3)"), ("6", "Fixed defect (6)"), ("7", "Reversible defect (7)")]
}

def build_form_values(source):
    values = {}
    for name in FEATURES:
        values[name] = source.get(name, "")
    return values

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    values = build_form_values(request.form if request.method == "POST" else {})
    if request.method == "POST":
        if model is None:
            error = "The model is not trained yet. Follow the README instructions and run: python train.py"
        else:
            try:
                row = []
                for name in FEATURES:
                    raw = request.form.get(name, "").strip()
                    if raw == "":
                        raise ValueError(f"Please fill in the {name} field.")
                    row.append(float(raw))
                # Basic range validation; category-specific options are shown in the form.
                limits = {
                    "age": (18, 100), "sex": (0, 1), "cp": (1, 4),
                    "trestbps": (70, 250), "chol": (80, 700), "fbs": (0, 1),
                    "restecg": (0, 2), "thalach": (50, 250), "exang": (0, 1),
                    "oldpeak": (0, 10), "slope": (1, 3), "ca": (0, 3), "thal": (3, 7)
                }
                for name, val in zip(FEATURES, row):
                    lo, hi = limits[name]
                    if not lo <= val <= hi:
                        raise ValueError(f"{name} must be between {lo} and {hi}.")
                    if name in {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"} and not val.is_integer():
                        raise ValueError(f"{name} must be a whole-number category.")
                prediction = int(model.predict([row])[0])
                score = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([row])[0]
                    classes = list(model.classes_)
                    if 1 in classes:
                        score = round(float(probs[classes.index(1)]) * 100, 1)
                result = {
                    "label": "Model flagged possible presence" if prediction == 1 else "Model did not flag presence",
                    "positive": prediction == 1,
                    "score": score
                }
            except (ValueError, TypeError) as exc:
                error = str(exc)
    return render_template("index.html", options=OPTIONS, values=values, result=result, error=error, model_ready=model is not None)

@app.route("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}

if __name__ == "__main__":
    app.run(debug=True)
