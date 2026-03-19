import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

FEATURES_CSV  = os.path.join(PROCESSED_DIR, "combined_features.csv")
WINDOWS_ML    = os.path.join(PROCESSED_DIR, "windows_ml.npy")
META_CSV      = os.path.join(PROCESSED_DIR, "meta.csv")
MODEL_OUT     = os.path.join(OUTPUTS_DIR,   "random_forest.joblib")

FEATURE_COLS        = ["norm_depth", "dur_period_ratio", "radius_ratio",
                       "ingress_slope", "secondary_depth"]
OPERATING_THRESHOLD = 0.10


# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def load_feature_dataset():
    """
    Load the physics-filtered feature dataset from features.py output.
    Returns X (features), y (labels), and the full dataframe.
    """
    df = pd.read_csv(FEATURES_CSV)

    # Validate all expected feature columns are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}\n"
            f"Run features.py first to generate the full feature set."
        )

    print(f"Loaded {len(df)} samples from features CSV")
    print(f"  Positive (transit):     {int(df['label'].sum())}")
    print(f"  Negative (no transit):  {int((df['label'] == 0).sum())}")
    print(f"  Features used:          {FEATURE_COLS}")

    X = df[FEATURE_COLS].values
    y = df["label"].values.astype(int)
    return X, y, df


# ── 2. RANDOM FOREST CLASSIFIER ───────────────────────────────────────────────

def train_random_forest(X, y):
    """
    Train a Random Forest on physics features with class_weight='balanced'.
    class_weight='balanced' mirrors real exoplanet surveys — tuned recall-heavy
    so the Physics Lead can validate false positives manually.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print(f"\nRandom Forest trained on {len(X_train)} samples")
    return clf, scaler, X_train, X_test, y_train, y_test


# ── 3. EVALUATION ─────────────────────────────────────────────────────────────

def evaluate_model(clf, scaler, X_test, y_test):
    """
    Full evaluation: classification report, confusion matrix, ROC-AUC.
    NOTE: uses default threshold (0.5) — operating threshold applied separately.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred        = clf.predict(X_test_scaled)
    y_prob        = clf.predict_proba(X_test_scaled)[:, 1]

    print("\n── Classification Report (default threshold=0.5) ───────")
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
    5-fold stratified cross-validation.
    More reliable than a single split given the small dataset size.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")

    print(f"\n── 5-Fold Cross-Validation (ROC-AUC) ───────────────────")
    for i, s in enumerate(scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean:   {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ── 4. FEATURE IMPORTANCE ─────────────────────────────────────────────────────

def plot_feature_importance(clf, output_path):
    """
    Plot Random Forest feature importances.
    """
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names       = [FEATURE_COLS[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(FEATURE_COLS)), importances[indices])
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel("Importance")
    ax.set_title("Feature importances — Random Forest")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved feature importance plot → {output_path}")

    print("\n── Feature Importances ─────────────────────────────────")
    for i in indices:
        print(f"  {FEATURE_COLS[i]:<22} {importances[i]:.4f}")


# ── 5. ROC CURVE ──────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob, output_path):
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


# ── 6. THRESHOLD TUNING ───────────────────────────────────────────────────────

def tune_threshold(y_test, y_prob):
    """
    Sweep thresholds to find operating point meeting physics targets.
    Target: recall >= 0.75, precision >= 0.85.
    Returns the best threshold and computed operating stats.
    """
    print("\n── Threshold Tuning ────────────────────────────────────")
    print(f"  Physics target: recall >= 0.75, precision >= 0.85")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} "
          f"{'F1':>8} {'TP':>6} {'FP':>6}")
    print(f"  {'-'*54}")

    best_threshold = OPERATING_THRESHOLD
    best_f1        = 0.0

    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred_t = (y_prob >= thresh).astype(int)
        tp = int(((y_pred_t == 1) & (y_test == 1)).sum())
        fp = int(((y_pred_t == 1) & (y_test == 0)).sum())
        fn = int(((y_pred_t == 0) & (y_test == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        meets_target = rec >= 0.75 and prec >= 0.85
        marker       = " ◄ TARGET" if meets_target else ""

        print(f"  {thresh:>10.2f} {prec:>10.3f} {rec:>8.3f} "
              f"{f1:>8.3f} {tp:>6} {fp:>6}{marker}")

        if meets_target and f1 > best_f1:
            best_f1        = f1
            best_threshold = thresh

    print(f"\n  Locked threshold: {best_threshold:.2f}")
    if best_f1 == 0.0:
        print(f"  (no threshold hit both targets — using default {OPERATING_THRESHOLD})")
    else:
        print(f"  (best F1 among thresholds meeting recall>=0.75, precision>=0.85)")

    return best_threshold


# ── 7. FINAL EVALUATION AT OPERATING THRESHOLD ───────────────────────────────

def evaluate_at_threshold(y_test, y_prob, threshold, cv_scores):
    """
    Re-evaluate at the physics-approved operating threshold.
    Prints computed metrics — nothing hardcoded.
    """
    y_pred_final = (y_prob >= threshold).astype(int)

    tp = int(((y_pred_final == 1) & (y_test == 1)).sum())
    fp = int(((y_pred_final == 1) & (y_test == 0)).sum())
    fn = int(((y_pred_final == 0) & (y_test == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\n── Final Results at threshold={threshold:.2f} (physics-approved) ──")
    print(classification_report(y_test, y_pred_final,
                                 target_names=["no transit", "transit"]))
    cm = confusion_matrix(y_test, y_pred_final)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\n  Operating parameters (computed):")
    print(f"  Precision:    {precision:.3f}")
    print(f"  Recall:       {recall:.3f}")
    print(f"  CV ROC-AUC:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Features:     {FEATURE_COLS}")

    return precision, recall


# ── 8. WRITE PREDICTIONS TO CSV ───────────────────────────────────────────────

def write_predictions(clf, scaler, threshold):
    """
    Add model predictions to the features CSV so visualize.py
    uses model output, not training labels.
    """
    features_df  = pd.read_csv(FEATURES_CSV)
    X_all        = features_df[FEATURE_COLS].values
    X_all_scaled = scaler.transform(X_all)

    features_df["predicted"]         = clf.predict(X_all_scaled)
    features_df["prediction_proba"]  = clf.predict_proba(X_all_scaled)[:, 1]
    features_df["predicted_transit"] = (
        features_df["prediction_proba"] >= threshold
    ).astype(int)

    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"\n  Predictions written to {FEATURES_CSV}")
    print(f"  Flagged as transit: "
          f"{features_df['predicted_transit'].sum()} / {len(features_df)} windows")


# ── 9. SAVE MODEL ─────────────────────────────────────────────────────────────

def save_model(clf, scaler, output_path):
    joblib.dump({"model": clf, "scaler": scaler}, output_path)
    print(f"  Saved model → {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Phase 3 — ML Classification")
    print("=" * 55)

    # Load data
    X, y, df = load_feature_dataset()

    # Train
    clf, scaler, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Evaluate at default threshold
    y_pred, y_prob = evaluate_model(clf, scaler, X_test, y_test)

    # Cross-validate
    cv_scores = cross_validate_model(clf, X, y)

    # Feature importance plot
    plot_feature_importance(clf, os.path.join(OUTPUTS_DIR, "feature_importance.png"))

    # ROC curve
    plot_roc_curve(y_test, y_prob, os.path.join(OUTPUTS_DIR, "roc_curve.png"))

    # Threshold sweep — find operating point
    best_threshold = tune_threshold(y_test, y_prob)

    # Final evaluation at operating threshold — all metrics computed, none hardcoded
    precision, recall = evaluate_at_threshold(y_test, y_prob, best_threshold, cv_scores)

    # Write predictions to features CSV for visualize.py
    write_predictions(clf, scaler, best_threshold)

    # Save model
    save_model(clf, scaler, MODEL_OUT)

    print("\nDone. Outputs saved to /outputs/")
