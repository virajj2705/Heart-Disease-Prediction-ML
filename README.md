# Heart Disease Prediction (Educational ML Project)

A small Flask website that demonstrates a binary classification model trained on the UCI Machine Learning Repository's Heart Disease dataset. It uses Python, Flask, scikit-learn, HTML, and CSS. **No database is required.**

> **Important:** This is a student learning project, not a medical device. Predictions and displayed model scores are not a diagnosis, clinical risk estimate, or substitute for a qualified healthcare professional. Do not use this app to make medical decisions.

## Features
- Modern responsive form and results page
- Random Forest classifier using scikit-learn
- Handles missing values in the training dataset with median imputation
- Input validation
- No login or database
- `/health` endpoint to check whether the model is ready

## Project structure

```text
heart_disease_prediction/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── heart_model.pkl              # created by train.py
├── heart_disease_dataset.zip    # downloaded by train.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Requirements
- Python 3.10–3.12 recommended
- VS Code
- Internet connection for the first dataset download

## Run in VS Code (Windows)

1. Extract the ZIP file.
2. Open the extracted `heart_disease_prediction` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run:

   ```bat
   .venv\Scripts\activate.bat
   ```

6. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Train the model and download the UCI dataset:

   ```powershell
   python train.py
   ```

   This creates `heart_model.pkl`. The first run needs an internet connection. Dataset source: [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

8. Start the website:

   ```powershell
   python app.py
   ```

9. Open this address in your browser:

   `http://127.0.0.1:5000`

10. Stop the server with `Ctrl+C` in the terminal.

## Input fields and encoding

The app uses the 13 Cleveland dataset features:

- `age`: age in years
- `sex`: 1 male, 0 female (dataset encoding)
- `cp`: chest-pain category, 1–4
- `trestbps`: resting blood pressure (mm Hg)
- `chol`: serum cholesterol (mg/dL)
- `fbs`: fasting blood sugar > 120 mg/dL, 1 yes / 0 no
- `restecg`: resting ECG category, 0–2
- `thalach`: maximum heart rate recorded
- `exang`: exercise-induced angina, 1 yes / 0 no
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: ST segment slope category, 1–3
- `ca`: number of major vessels (0–3)
- `thal`: thal category encoded as 3, 6, or 7

The categories are historical dataset codes, not a recommendation to self-assess or interpret medical findings. The app accepts values for a classroom demonstration.

## Troubleshooting

- **`python` not found:** install Python from [python.org](https://www.python.org/downloads/) and enable “Add Python to PATH”, then reopen VS Code.
- **Dataset download fails:** check your internet connection. Delete a partially downloaded `heart_disease_dataset.zip` and rerun `python train.py`.
- **Model not trained:** run `python train.py` before `python app.py`.
- **Port 5000 busy:** change `app.run(debug=True)` to `app.run(debug=True, port=5001)` and open `http://127.0.0.1:5001`.
- **PowerShell activation blocked:** use the Command Prompt activation command shown above.

## Academic note
The model is trained on a small, older, limited dataset and may be biased or inaccurate for people outside that dataset. The probability output is a model score, not a calibrated estimate of an individual's real-world probability of heart disease.
