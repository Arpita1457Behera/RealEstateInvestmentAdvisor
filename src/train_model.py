from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "housing_model_data.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "property_price_model.joblib"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}\n"
            "Run src/data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH)

    # Ensure the target is numeric.
    df["Price_in_Lakhs"] = pd.to_numeric(
        df["Price_in_Lakhs"], errors="coerce"
    )

    # Keep rows with valid target values.
    df = df.dropna(subset=["Price_in_Lakhs"])
    df = df[df["Price_in_Lakhs"] > 0].copy()

    return df


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

def main():
    df = load_data()

    # Price_in_Lakhs is the value we want to predict.
    y = df["Price_in_Lakhs"]

    # Exclude target and price-derived columns to prevent leakage.
    excluded_columns = [
        "ID",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Calculated_Price_per_SqFt_Rs",
    ]

    X = df.drop(
        columns=excluded_columns,
        errors="ignore",
    )

    # Remove columns that contain no useful variation.
    constant_columns = [
        column for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]

    X = X.drop(columns=constant_columns)

    # Identify numeric and categorical features.
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print("=" * 55)
    print("PROPERTY PRICE PREDICTION - MODEL TRAINING")
    print("=" * 55)

    print("\nDataset rows:", len(df))
    print("Number of features:", len(X.columns))
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Split before fitting preprocessing, so the test data
    # does not influence imputation or category encoding.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )

    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 55)
    print("MODEL EVALUATION")
    print("=" * 55)

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"MAE:  ₹{mae:.2f} lakhs")
    print(f"RMSE: ₹{rmse:.2f} lakhs")
    print(f"R² score: {r2:.4f}")

    # Save the trained model.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
