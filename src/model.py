import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

FEATURES_CSV  = os.path.join(PROCESSED_DIR, "combined_features.csv")
WINDOWS_ML    = os.path.join(PROCESSED_DIR, "KIC_11446443", "windows_ml.npy")
META_CSV      = os.path.join(PROCESSED_DIR, "KIC_11446443", "meta.csv")
MODEL_OUT     = os.path.join(OUTPUTS_DIR,   "random_forest.joblib")

FEATURE_COLS  = ["norm_depth", "dur_period_ratio", "radius_ratio"]


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def load_feature_dataset():
    """
    Load the physics-filtered feature dataset from features.py output.
    Returns X (features), y (labels), and the full dataframe.
    """
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} samples from features CSV")
    print(f"  Positive (transit):     {int(df['label'].sum())}")
    print(f"  Negative (no transit):  {int((df['label'] == 0).sum())}")

    X = df[FEATURE_COLS].values
    y = df["label"].values.astype(int)
    return X, y, df


def load_window_dataset():
    """
    Load raw windowed light curves + labels for CNN input.
    Merges windows_ml.npy with meta.csv labels.
    """
    windows = np.load(WINDOWS_ML)
    meta    = pd.read_csv(META_CSV)

    # Only keep windows that passed the physics filter
    features_df = pd.read_csv(FEATURES_CSV)
    valid_idx   = features_df["window_index"].values.astype(int)

    X = windows[valid_idx]
    y = meta.iloc[valid_idx]["label"].values.astype(int)

    print(f"Loaded {len(X)} windows of shape {X.shape[1]}")
    print(f"  Positive (transit):    {int(y.sum())}")
    print(f"  Negative (no transit): {int((y == 0).sum())}")
    return X, y


# ── 2. RANDOM FOREST CLASSIFIER ───────────────────────────────────────────────

def train_random_forest(X, y):
    """
    Train a Random Forest on physics features with class_weight='balanced'
    to handle the transit/non-transit class imbalance (~3.5% positive rate).

    class_weight='balanced' penalizes missing a transit more heavily than
    a false positive — correct physics choice since missed planets are worse
    than false alarms.
    """
    # Stratified split preserves class ratio in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",   # physics-approved imbalance fix
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print(f"\nRandom Forest trained on {len(X_train)} samples")
    return clf, scaler, X_train, X_test, y_train, y_test


# ── 3. EVALUATION ─────────────────────────────────────────────────────────────

def evaluate_model(clf, scaler, X_test, y_test):
    """
    Full evaluation: classification report, confusion matrix,
    ROC-AUC, and cross-validation score.
    """
    X_test_scaled = scaler.transform(X_test) if not hasattr(scaler, '_fitted') \
                    else X_test
    y_pred        = clf.predict(X_test_scaled)
    y_prob        = clf.predict_proba(X_test_scaled)[:, 1]

    print("\n── Classification Report ───────────────────────────────")
    print(classification_report(y_test, y_pred,
                                 target_names=["no transit", "transit"]))

    print("── Confusion Matrix ────────────────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\n  ROC-AUC: {roc_auc:.4f}")
    else:
        print("\n  ROC-AUC: N/A (only one class in test set)")

    return y_pred, y_prob


def cross_validate_model(clf, X, y):
    """
    5-fold stratified cross-validation — more reliable than a single split
    given the small dataset size.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores   = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

    print(f"\n── 5-Fold Cross-Validation (ROC-AUC) ───────────────────")
    for i, s in enumerate(scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean:   {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ── 4. FEATURE IMPORTANCE ─────────────────────────────────────────────────────

def plot_feature_importance(clf, output_path):
    """
    Plot Random Forest feature importances.
    Tells us which physics features matter most for transit detection.
    """
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names       = [FEATURE_COLS[i] for i in indices]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(FEATURE_COLS)), importances[indices])
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Importance")
    ax.set_title("Feature importances — Random Forest")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved feature importance plot → {output_path}")

    print("\n── Feature Importances ─────────────────────────────────")
    for i in indices:
        print(f"  {FEATURE_COLS[i]:<20} {importances[i]:.4f}")


# ── 5. ROC CURVE ──────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob, output_path):
    """
    Plot ROC curve with AUC score.
    """
    if len(np.unique(y_test)) < 2:
        print("  Skipping ROC plot — only one class in test set")
        return

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc         = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — transit detection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved ROC curve → {output_path}")


# ── 6. SAVE MODEL ─────────────────────────────────────────────────────────────

def save_model(clf, scaler, output_path):
    joblib.dump({"model": clf, "scaler": scaler}, output_path)
    print(f"  Saved model → {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Phase 3 — ML Classification")
    print("=" * 55)

    # Load physics-filtered features
    X, y, df = load_feature_dataset()

    # Train Random Forest
    clf, scaler, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Evaluate on held-out test set
    y_pred, y_prob = evaluate_model(clf, scaler, X_test, y_test)

    # Cross-validate for reliable performance estimate
    cross_validate_model(clf, X, y)

    # Feature importance plot
    plot_feature_importance(
        clf,
        os.path.join(OUTPUTS_DIR, "feature_importance.png")
    )

    # ROC curve
    plot_roc_curve(
        y_test, y_prob,
        os.path.join(OUTPUTS_DIR, "roc_curve.png")
    )

    # Save model
    save_model(clf, scaler, MODEL_OUT)

    print("\nDone. Outputs saved to /outputs/")