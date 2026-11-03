from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "india_housing_prices.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "housing_model_data.csv"


def load_data():
    """Load the original housing dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


def preprocess_data(df):
    """Validate and prepare a separate copy of the dataset."""
    data = df.copy()

    # Remove duplicate rows, if any.
    data = data.drop_duplicates()

    # Remove duplicate IDs, keeping the first occurrence.
    if "ID" in data.columns:
        data = data.drop_duplicates(subset=["ID"], keep="first")

    # Ensure essential numeric columns contain valid values.
    numeric_columns = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Year_Built",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    # Remove records with invalid essential values.
    essential_columns = [
        "Size_in_SqFt",
        "Price_in_Lakhs",
    ]

    data = data.dropna(subset=essential_columns)

    data = data[
        (data["Size_in_SqFt"] > 0)
        & (data["Price_in_Lakhs"] > 0)
    ].copy()

    # Calculate price per square foot in rupees.
    data["Calculated_Price_per_SqFt_Rs"] = (
        data["Price_in_Lakhs"] * 100000
        / data["Size_in_SqFt"]
    )

    # Floor fields have substantial inconsistencies.
    # Exclude them from the initial modelling dataset.
    data = data.drop(
        columns=["Floor_No", "Total_Floors"],
        errors="ignore",
    )

    return data


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    original_data = load_data()
    cleaned_data = preprocess_data(original_data)

    cleaned_data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print("=" * 50)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 50)
    print("Original rows:", len(original_data))
    print("Processed rows:", len(cleaned_data))
    print("Processed columns:", len(cleaned_data.columns))
    print("Saved to:", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
