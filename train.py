from pathlib import Path
from io import BytesIO
import urllib.request
import zipfile
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ZIP_PATH = BASE_DIR / "heart_disease_dataset.zip"
MODEL_PATH = BASE_DIR / "heart_model.pkl"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def get_dataset():
    # Download the official UCI Heart Disease archive if not already present.
    if not ZIP_PATH.exists():
        print("Downloading the UCI Heart Disease dataset...")
        request = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            ZIP_PATH.write_bytes(response.read())

    with zipfile.ZipFile(ZIP_PATH) as archive:
        candidates = [n for n in archive.namelist() if n.lower().endswith("processed.cleveland.data")]
        if not candidates:
            raise FileNotFoundError(
                "Could not find processed.cleveland.data in the downloaded archive. "
                "Delete heart_disease_dataset.zip and try again."
            )
        with archive.open(candidates[0]) as raw:
            df = pd.read_csv(raw, header=None, names=COLUMNS, na_values="?")
    # Cleveland target: 0 means no diagnosed disease; 1–4 are presence levels.
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"])
    df["target"] = (df["target"] > 0).astype(int)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target"])
    return df

def main():
    df = get_dataset()
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(f"Dataset rows used: {len(df)}")
    print(f"Holdout accuracy (educational benchmark only): {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    with MODEL_PATH.open("wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nSaved model to: {MODEL_PATH}")
    print("This model is for learning/demo use, not clinical decision-making.")

if __name__ == "__main__":
    main()
