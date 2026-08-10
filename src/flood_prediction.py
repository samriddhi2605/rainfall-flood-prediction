"""
Rainfall-Based Flood Risk Prediction
====================================

Educational machine-learning project using public IMD rainfall data.

The model classifies historical Kerala monsoon seasons into:
0 = Normal rainfall-risk class
1 = Higher rainfall-risk class

Important:
This is NOT an operational flood-warning system. The target is a
rainfall-based proxy created from the historical distribution.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "rainfall_data.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "flood_risk_model.joblib"

RANDOM_STATE = 42

# The top 25% of historical monsoon rainfall is used as a
# high-rainfall-risk proxy. This is a project-defined label,
# not an official flood threshold.
RISK_PERCENTILE = 75


# ------------------------------------------------------------
# Load and clean data
# ------------------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    required = {
        "SUBDIVISION",
        "YEAR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "JJAS",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    # Convert rainfall columns to numeric.
    rainfall_cols = [
        "MAY", "JUN", "JUL", "AUG", "SEP", "JJAS"
    ]

    for col in rainfall_cols:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df["YEAR"] = pd.to_numeric(
        df["YEAR"], errors="coerce"
    )

    # Keep Kerala subdivision.
    kerala = df[
        df["SUBDIVISION"]
        .astype(str)
        .str.upper()
        .str.contains("KERALA")
    ].copy()

    kerala = kerala.dropna(
        subset=[
            "YEAR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "JJAS",
        ]
    )

    if kerala.empty:
        raise ValueError(
            "No usable Kerala records were found."
        )

    return kerala


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

def engineer_features(df):
    data = df.copy()

    # Use the dataset's JJAS value as the monsoon total.
    data["monsoon_rainfall"] = data["JJAS"]

    # May rainfall before the main monsoon.
    data["pre_monsoon"] = data["MAY"]

    # Early monsoon rainfall.
    data["early_monsoon"] = (
        data["JUN"] + data["JUL"]
    )

    # Peak monsoon period.
    data["peak_monsoon"] = (
        data["JUL"] + data["AUG"]
    )

    # Late monsoon rainfall.
    data["late_monsoon"] = (
        data["AUG"] + data["SEP"]
    )

    # Change from May to June.
    data["may_june_change"] = (
        data["JUN"] - data["MAY"]
    )

    # Variation among the four monsoon months.
    data["monsoon_variability"] = data[
        ["JUN", "JUL", "AUG", "SEP"]
    ].std(axis=1)

    # June's contribution to the monsoon.
    data["june_share"] = (
        data["JUN"]
        / data["JJAS"].replace(0, pd.NA)
    )

    return data


# ------------------------------------------------------------
# Create project target
# ------------------------------------------------------------

def create_target(df):
    data = df.copy()

    threshold = data[
        "monsoon_rainfall"
    ].quantile(RISK_PERCENTILE / 100)

    data["high_rainfall_risk"] = (
        data["monsoon_rainfall"]
        >= threshold
    ).astype(int)

    return data, float(threshold)


FEATURES = [
    "pre_monsoon",
    "early_monsoon",
    "peak_monsoon",
    "late_monsoon",
    "may_june_change",
    "monsoon_variability",
    "june_share",
]

TARGET = "high_rainfall_risk"


# ------------------------------------------------------------
# Build models
# ------------------------------------------------------------

def build_models():
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_split=4,
        random_state=RANDOM_STATE,
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
    }


# ------------------------------------------------------------
# Train and compare
# ------------------------------------------------------------

def train_and_compare(X_train, X_test, y_train, y_test):
    models = build_models()

    trained = {}
    scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        score = accuracy_score(
            y_test,
            prediction
        )

        trained[name] = model
        scores[name] = score

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        print(f"Test accuracy: {score:.2%}")
        print(
            classification_report(
                y_test,
                prediction,
                zero_division=0,
            )
        )

    best_name = max(
        scores,
        key=scores.get
    )

    return (
        trained[best_name],
        best_name,
        scores,
    )


# ------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------

def save_rainfall_trend(data):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        data["YEAR"],
        data["monsoon_rainfall"],
        marker=".",
        linewidth=1,
    )

    plt.title(
        "Kerala Monsoon Rainfall Over Time"
    )

    plt.xlabel("Year")
    plt.ylabel("JJAS Rainfall (mm)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "rainfall_trend.png",
        dpi=200,
    )

    plt.close()


def save_confusion_matrix(model, X_test, y_test):
    prediction = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        prediction,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Normal",
            "Higher Risk",
        ],
    )

    display.plot()

    plt.title(
        "High-Rainfall Risk Classification"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
    )

    plt.close()


def save_feature_importance(model):
    if not hasattr(
        model,
        "feature_importances_"
    ):
        return

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values()

    importance.plot(
        kind="barh",
        figsize=(9, 5),
    )

    plt.title(
        "Rainfall Feature Importance"
    )

    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=200,
    )

    plt.close()


# ------------------------------------------------------------
# Custom prediction
# ------------------------------------------------------------

def predict_risk(model, rainfall):
    """
    Example input:
    {
        "MAY": 100,
        "JUN": 600,
        "JUL": 800,
        "AUG": 700,
        "SEP": 300
    }
    """

    may = rainfall["MAY"]
    june = rainfall["JUN"]
    july = rainfall["JUL"]
    august = rainfall["AUG"]
    september = rainfall["SEP"]

    monsoon = (
        june + july + august + september
    )

    row = pd.DataFrame([{
        "pre_monsoon": may,
        "early_monsoon": june + july,
        "peak_monsoon": july + august,
        "late_monsoon": august + september,
        "may_june_change": june - may,
        "monsoon_variability": pd.Series(
            [june, july, august, september]
        ).std(),
        "june_share": (
            june / monsoon
            if monsoon else 0
        ),
    }])

    prediction = int(
        model.predict(row)[0]
    )

    probability = float(
        model.predict_proba(row)[0][1]
    )

    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return prediction, probability, risk


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("RAINFALL-BASED FLOOD RISK PREDICTION")
    print("=" * 60)

    data = load_data()

    print(
        f"\nKerala records: {len(data)}"
    )

    data = engineer_features(data)

    data, threshold = create_target(data)

    data = data.dropna(
        subset=FEATURES + [TARGET]
    )

    X = data[FEATURES]
    y = data[TARGET]

    print(
        f"High-risk rainfall threshold: "
        f"{threshold:.2f} mm"
    )

    print("\nClass distribution:")
    print(y.value_counts())

    # A chronological split is more realistic for time-series-like
    # historical rainfall. The final 20% of years are held out.
    data = data.sort_values("YEAR").reset_index(drop=True)

    split_index = int(len(data) * 0.80)

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]

    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    best_model, best_name, scores = train_and_compare(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    print(f"Test accuracy: {scores[best_name]:.2%}")
    print("=" * 60)

    # Cross-validation is performed on the training period only.
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    print(
        f"\n5-fold training CV accuracy: "
        f"{cv_scores.mean():.2%}"
    )

    save_rainfall_trend(data)
    save_confusion_matrix(
        best_model,
        X_test,
        y_test,
    )

    if best_name == "Random Forest":
        save_feature_importance(
            best_model
        )

    joblib.dump(
        {
            "model": best_model,
            "features": FEATURES,
            "threshold": threshold,
            "risk_percentile": RISK_PERCENTILE,
        },
        MODEL_PATH,
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        "\nOutput files are available in:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
