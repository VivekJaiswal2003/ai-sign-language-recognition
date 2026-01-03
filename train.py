"""
Train ASL static sign classifier from `asl_landmark_dataset.csv`.
"""

from pathlib import Path
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Allowed classes
ALLOWED_CLASSES = [
     'HELLO', 'GOOD MORNING TO ALL', 'HAVE A NICE DAY','RESCUE',
    'HELP', 'GOOD NIGHT', 'I AM VIVEK KUMAR', 'I AM VINAY KUMAR','I AM TUSHAR SHARMA','I AM VINIT',
    'I AM VISHAL', 'NEED PEN', 'HOW ARE YOU SIR', 'FIVE GROUP MEMBERS', 'I AM HUNGURY',
]

# -----------------------------
# LOAD & FILTER DATA
# -----------------------------
def load_and_filter(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'class_label' not in df.columns:
        raise ValueError("Input CSV must contain class_label")

    initial = len(df)
    df = df[df['class_label'].isin(ALLOWED_CLASSES)]
    print(f"📥 Loaded: {initial}, Kept: {len(df)}, Removed: {initial-len(df)}")
    return df


# -----------------------------
# FEATURE PREP
# -----------------------------
def prepare_features(df):
    X = df.drop(columns=['class_label']).values
    y = df['class_label'].values

    if X.shape[1] != 63:
        warnings.warn("Expected 63 features")

    mask = np.isfinite(X).all(axis=1)
    return X[mask], y[mask]


# -----------------------------
# NORMALIZATION
# -----------------------------
def _normalize_sample(landmarks):
    arr = landmarks.reshape(-1, 3).astype(float)
    wrist = arr[0].copy()

    arr[:, :2] -= wrist[:2]
    arr[:, 2] -= wrist[2]

    dists = np.linalg.norm(arr[:, :2], axis=1)
    scale = dists.max() if dists.max() > 1e-6 else 1.0

    arr[:, :2] /= scale
    arr[:, 2] /= scale
    return arr.flatten()

def normalize_dataset(X):
    Xn = np.empty_like(X)
    for i in range(X.shape[0]):
        try:
            Xn[i] = _normalize_sample(X[i])
        except:
            Xn[i] = X[i]
    return Xn


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):

    from xgboost import XGBClassifier

    # Label Encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n⚙ Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=700,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='mlogloss'
    )

    xgb.fit(X_train_scaled, y_train_enc)

    pred_enc = xgb.predict(X_test_scaled)
    pred = le.inverse_transform(pred_enc)

    acc = accuracy_score(y_test, pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

    # Save encoder for detect.py
    joblib.dump(le, "asl_label_encoder.pkl")

    return scaler, {'xgb': {'model': xgb, 'accuracy': acc}}


# -----------------------------
# SAVE MODEL
# -----------------------------
def pick_and_save_best(results, scaler, model_path, scaler_path):

    best_model = results['xgb']['model']
    best_acc = results['xgb']['accuracy']

    print(f"\n🏁 Best Model: XGBoost ({best_acc:.4f})")

    joblib.dump(best_model, model_path)
    print("💾 Saved model:", model_path)

    joblib.dump(scaler, scaler_path)
    print("💾 Saved scaler:", scaler_path)

    return best_acc


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = argparse.ArgumentParser().parse_args()

    df = load_and_filter(Path("asl_landmark_dataset.csv"))
    X, y = prepare_features(df)

    print("🔧 Normalizing…")
    X = normalize_dataset(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler, results = train_and_evaluate(X_train, X_test, y_train, y_test)

    pick_and_save_best(results, scaler,
                       Path("asl_sign_model.pkl"),
                       Path("asl_scaler.pkl"))

if __name__ == "__main__":
    main()
