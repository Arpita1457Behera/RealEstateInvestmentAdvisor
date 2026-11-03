from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "india_housing_prices.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}\n"
        "Place india_housing_prices.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print("\nColumn names:")
print(df.columns.tolist())

print("\n========== DATA QUALITY ==========")
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

if "ID" in df.columns:
    print("Duplicate IDs:", int(df["ID"].duplicated().sum()))

# Save data audit
audit = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(df[c].dtype) for c in df.columns],
    "Missing_Values": [int(df[c].isna().sum()) for c in df.columns],
    "Unique_Values": [int(df[c].nunique()) for c in df.columns]
})

audit.to_csv(OUTPUT_DIR / "data_audit.csv", index=False)

# --------------------------------------------------
# 3. DERIVED PRICE FEATURE
# --------------------------------------------------
# Price_in_Lakhs is assumed to be measured in lakh rupees.
# The calculation converts it into rupees per square foot.

if {"Price_in_Lakhs", "Size_in_SqFt"}.issubset(df.columns):
    df["Calculated_Price_per_SqFt_Rs"] = (
        df["Price_in_Lakhs"] * 100000
        / df["Size_in_SqFt"].replace(0, np.nan)
    )


# --------------------------------------------------
# 4. HELPER FUNCTION FOR CHARTS
# --------------------------------------------------

def save_chart(filename, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# Q1. PROPERTY PRICE DISTRIBUTION
# --------------------------------------------------

if "Price_in_Lakhs" in df:
    df["Price_in_Lakhs"].dropna().hist(bins=40)
    save_chart(
        "01_price_distribution.png",
        "Property Price Distribution",
        "Price (₹ lakhs)",
        "Number of Properties"
    )


# --------------------------------------------------
# Q2. PROPERTY SIZE DISTRIBUTION
# --------------------------------------------------

if "Size_in_SqFt" in df:
    df["Size_in_SqFt"].dropna().hist(bins=40)
    save_chart(
        "02_size_distribution.png",
        "Property Size Distribution",
        "Size (sq ft)",
        "Number of Properties"
    )


# --------------------------------------------------
# Q3. PRICE PER SQ FT BY PROPERTY TYPE
# --------------------------------------------------

if {"Property_Type", "Calculated_Price_per_SqFt_Rs"}.issubset(df.columns):
    groups = df.groupby("Property_Type", observed=True)

    values = [
        group["Calculated_Price_per_SqFt_Rs"].dropna().values
        for _, group in groups
    ]
    labels = [str(name) for name in groups.groups.keys()]

    if values:
        plt.figure(figsize=(8, 5))
        plt.boxplot(values, tick_labels=labels, showfliers=False)
        save_chart(
            "03_price_sqft_by_type.png",
            "Calculated Price per Sq Ft by Property Type",
            "Property Type",
            "Price (₹/sq ft)"
        )


# --------------------------------------------------
# Q4. PROPERTY SIZE VS PRICE
# --------------------------------------------------

if {"Size_in_SqFt", "Price_in_Lakhs"}.issubset(df.columns):
    sample = df[["Size_in_SqFt", "Price_in_Lakhs"]].dropna()

    if len(sample) > 10000:
        sample = sample.sample(10000, random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        sample["Size_in_SqFt"],
        sample["Price_in_Lakhs"],
        alpha=0.35,
        s=10
    )

    save_chart(
        "04_size_vs_price.png",
        "Property Size vs Price",
        "Size (sq ft)",
        "Price (₹ lakhs)"
    )


# --------------------------------------------------
# Q5. OUTLIER CHECKS
# --------------------------------------------------

for column, filename, title in [
    ("Size_in_SqFt", "05_size_outliers.png", "Property Size Outlier Check"),
    (
        "Calculated_Price_per_SqFt_Rs",
        "05_price_sqft_outliers.png",
        "Price per Sq Ft Outlier Check"
    )
]:
    if column in df:
        plt.figure(figsize=(8, 3))
        plt.boxplot(df[column].dropna(), vert=False, showfliers=False)

        save_chart(
            filename,
            title,
            column.replace("_", " "),
            ""
        )


# --------------------------------------------------
# Q6. AVERAGE PRICE PER SQ FT BY STATE
# --------------------------------------------------

if {"State", "Calculated_Price_per_SqFt_Rs"}.issubset(df.columns):
    values = (
        df.groupby("State", observed=True)["Calculated_Price_per_SqFt_Rs"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    values.plot(kind="barh")

    save_chart(
        "06_price_sqft_by_state.png",
        "Average Calculated Price per Sq Ft by State",
        "Price (₹/sq ft)",
        "State"
    )


# --------------------------------------------------
# Q7. AVERAGE PROPERTY PRICE BY CITY
# --------------------------------------------------

if {"City", "Price_in_Lakhs"}.issubset(df.columns):
    values = (
        df.groupby("City", observed=True)["Price_in_Lakhs"]
        .mean()
        .nlargest(15)
        .sort_values()
    )

    plt.figure(figsize=(9, 6))
    values.plot(kind="barh")

    save_chart(
        "07_price_by_city.png",
        "15 Cities with Highest Average Property Price",
        "Average Price (₹ lakhs)",
        "City"
    )


# --------------------------------------------------
# Q8. MEDIAN PROPERTY AGE BY LOCALITY
# --------------------------------------------------

if {"Locality", "Age_of_Property"}.issubset(df.columns):
    values = (
        df.groupby("Locality", observed=True)["Age_of_Property"]
        .median()
        .nlargest(15)
        .sort_values()
    )

    plt.figure(figsize=(9, 6))
    values.plot(kind="barh")

    save_chart(
        "08_age_by_locality.png",
        "15 Localities with Highest Median Property Age",
        "Median Age (recorded years)",
        "Locality"
    )


# --------------------------------------------------
# Q9. BHK DISTRIBUTION ACROSS CITIES
# --------------------------------------------------

if {"City", "BHK"}.issubset(df.columns):
    top_cities = df["City"].value_counts().head(10).index
    subset = df[df["City"].isin(top_cities)]

    table = pd.crosstab(subset["City"], subset["BHK"])

    table.plot(kind="bar", stacked=True, figsize=(11, 6))

    save_chart(
        "09_bhk_by_city.png",
        "BHK Distribution Across the 10 Most Represented Cities",
        "City",
        "Number of Properties"
    )


# --------------------------------------------------
# Q10. TOP FIVE LOCALITIES BY AVERAGE PRICE
# --------------------------------------------------
# The dataset has no historical date series.
# This is a cross-sectional comparison, not a price trend.

if {"Locality", "Price_in_Lakhs"}.issubset(df.columns):
    values = (
        df.groupby("Locality", observed=True)["Price_in_Lakhs"]
        .mean()
        .nlargest(5)
        .sort_values()
    )

    plt.figure(figsize=(9, 4))
    values.plot(kind="barh")

    save_chart(
        "10_top_localities.png",
        "Top Five Localities by Average Price",
        "Average Price (₹ lakhs)",
        "Locality"
    )


# --------------------------------------------------
# Q11. NUMERIC FEATURE CORRELATIONS
# --------------------------------------------------

numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] > 1:
    correlation = numeric_df.corr(numeric_only=True)

    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(12, 9))

    plt.imshow(correlation, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        range(len(correlation.index)),
        correlation.index,
        fontsize=7
    )

    save_chart(
        "11_correlation_matrix.png",
        "Numeric Feature Correlation Matrix",
        "",
        ""
    )


# --------------------------------------------------
# Q12. NEARBY SCHOOLS VS PRICE PER SQ FT
# --------------------------------------------------

if {"Nearby_Schools", "Calculated_Price_per_SqFt_Rs"}.issubset(df.columns):
    values = (
        df.groupby("Nearby_Schools", observed=True)
        ["Calculated_Price_per_SqFt_Rs"]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(8, 4))
    values.plot(marker="o")

    save_chart(
        "12_schools_vs_price.png",
        "Nearby Schools vs Calculated Price per Sq Ft",
        "Nearby Schools",
        "Average Price (₹/sq ft)"
    )


# --------------------------------------------------
# Q13. NEARBY HOSPITALS VS PRICE PER SQ FT
# --------------------------------------------------

if {"Nearby_Hospitals", "Calculated_Price_per_SqFt_Rs"}.issubset(df.columns):
    values = (
        df.groupby("Nearby_Hospitals", observed=True)
        ["Calculated_Price_per_SqFt_Rs"]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(8, 4))
    values.plot(marker="o")

    save_chart(
        "13_hospitals_vs_price.png",
        "Nearby Hospitals vs Calculated Price per Sq Ft",
        "Nearby Hospitals",
        "Average Price (₹/sq ft)"
    )


# --------------------------------------------------
# Q14. PRICE BY FURNISHING STATUS
# --------------------------------------------------

if {"Furnished_Status", "Price_in_Lakhs"}.issubset(df.columns):
    values = (
        df.groupby("Furnished_Status", observed=True)["Price_in_Lakhs"]
        .median()
        .sort_values()
    )

    values.plot(kind="bar", figsize=(8, 4))

    save_chart(
        "14_furnishing_vs_price.png",
        "Median Price by Furnishing Status",
        "Furnishing Status",
        "Median Price (₹ lakhs)"
    )


# --------------------------------------------------
# Q15. PRICE PER SQ FT BY FACING
# --------------------------------------------------

if {"Facing", "Calculated_Price_per_SqFt_Rs"}.issubset(df.columns):
    values = (
        df.groupby("Facing", observed=True)
        ["Calculated_Price_per_SqFt_Rs"]
        .median()
        .sort_values()
    )

    values.plot(kind="bar", figsize=(8, 4))

    save_chart(
        "15_facing_vs_price.png",
        "Median Price per Sq Ft by Facing",
        "Facing",
        "Median Price (₹/sq ft)"
    )


# --------------------------------------------------
# Q16. PROPERTIES BY OWNER TYPE
# --------------------------------------------------

if "Owner_Type" in df:
    df["Owner_Type"].value_counts(dropna=False).plot(
        kind="bar",
        figsize=(8, 4)
    )

    save_chart(
        "16_owner_type.png",
        "Properties by Owner Type",
        "Owner Type",
        "Number of Properties"
    )


# --------------------------------------------------
# Q17. AVAILABILITY STATUS COUNTS
# --------------------------------------------------

if "Availability_Status" in df:
    df["Availability_Status"].value_counts(dropna=False).plot(
        kind="bar",
        figsize=(8, 4)
    )

    save_chart(
        "17_availability.png",
        "Properties by Availability Status",
        "Availability Status",
        "Number of Properties"
    )


# --------------------------------------------------
# Q18. PARKING VS PROPERTY PRICE
# --------------------------------------------------

if {"Parking_Space", "Price_in_Lakhs"}.issubset(df.columns):
    values = (
        df.groupby("Parking_Space", observed=True)["Price_in_Lakhs"]
        .median()
        .sort_values()
    )

    values.plot(kind="bar", figsize=(8, 4))

    save_chart(
        "18_parking_vs_price.png",
        "Median Property Price by Parking Availability",
        "Parking Space",
        "Median Price (₹ lakhs)"
    )


# --------------------------------------------------
# 19. AMENITIES VS PRICE PER SQ FT
# --------------------------------------------------

import ast

def count_amenities(value):
    if pd.isna(value):
        return 0

    text = str(value).strip()

    # Handle lists stored as strings, such as "['Gym', 'Garden']"
    if text.startswith("[") and text.endswith("]"):
        try:
            items = ast.literal_eval(text)
            return len(items) if isinstance(items, list) else 0
        except (ValueError, SyntaxError):
            pass

    # Handle comma-separated amenity lists
    if "," in text:
        return len([item for item in text.split(",") if item.strip()])

    # A single non-empty amenity
    return 1 if text else 0


df["Amenity_Count"] = df["Amenities"].apply(count_amenities)

amenity_summary = (
    df.groupby("Amenity_Count")["Calculated_Price_per_SqFt_Rs"]
    .median()
    .reset_index()
)

plt.figure(figsize=(10, 5))
plt.bar(
    amenity_summary["Amenity_Count"].astype(str),
    amenity_summary["Calculated_Price_per_SqFt_Rs"]
)

plt.title("Median Price per Sq Ft by Number of Amenities")
plt.xlabel("Number of Amenities")
plt.ylabel("Median Price (₹/sq ft)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "19_amenities_vs_price_sqft.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nMedian price per sq ft by amenity count:")
print(amenity_summary.to_string(index=False))

# --------------------------------------------------
# Q20. PUBLIC TRANSPORT VS PRICE PER SQ FT
# --------------------------------------------------

if {
    "Public_Transport_Accessibility",
    "Calculated_Price_per_SqFt_Rs"
}.issubset(df.columns):

    values = (
        df.groupby("Public_Transport_Accessibility", observed=True)
        ["Calculated_Price_per_SqFt_Rs"]
        .median()
        .sort_values()
    )

    values.plot(kind="bar", figsize=(8, 4))

    save_chart(
        "20_transport_vs_price.png",
        "Median Price per Sq Ft by Public Transport Accessibility",
        "Transport Accessibility",
        "Median Price (₹/sq ft)"
    )

# --------------------------------------------------
# OUTLIER COUNTS USING THE IQR METHOD
# --------------------------------------------------

print("\n========== OUTLIER ANALYSIS ==========")

for column in [
    "Size_in_SqFt",
    "Calculated_Price_per_SqFt_Rs",
]:
    if column not in df.columns:
        continue

    values = df[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_bound) |
        (values > upper_bound)
    ]

    print(f"\nColumn: {column}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Outlier count: {len(outliers):,}")
    print(f"Outlier percentage: {len(outliers) / len(values) * 100:.2f}%")

# --------------------------------------------------
# 5. SUMMARY TABLES AND QUALITY FLAGS
# --------------------------------------------------

df.select_dtypes(include="number").describe().T.to_csv(
    OUTPUT_DIR / "numeric_summary.csv"
)

if {"Floor_No", "Total_Floors"}.issubset(df.columns):
    invalid_floors = int((df["Floor_No"] > df["Total_Floors"]).sum())

    print(
        "\nRecords where Floor_No > Total_Floors:",
        f"{invalid_floors:,}"
    )

if {"Year_Built", "Age_of_Property"}.issubset(df.columns):
    matching_age = int(
        (df["Age_of_Property"] == 2025 - df["Year_Built"]).sum()
    )

    print(
        "Records where Age_of_Property equals 2025 - Year_Built:",
        f"{matching_age:,}"
    )

# Check the relationship between property size and price

correlation = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])

print("\n========== SIZE VS PRICE ==========")
print(f"Correlation: {correlation:.4f}")

if correlation > 0.5:
    print("Strong positive relationship.")
elif correlation > 0.3:
    print("Moderate positive relationship.")
elif correlation > 0:
    print("Weak positive relationship.")
elif correlation < -0.3:
    print("Negative relationship.")
else:
    print("Little linear relationship detected.")

print("\n========== EDA COMPLETE ==========")
print(f"Charts saved in: {CHART_DIR}")
print(f"Audit and summaries saved in: {OUTPUT_DIR}")

print(
    "\nImportant: This dataset does not contain verified investment labels "
    "or actual five-year future prices. Those modelling tasks require "
    "separate, clearly documented decisions."
)


# ==========================================
# DATA QUALITY VALIDATION
# ==========================================

print("\n" + "=" * 50)
print("DATA QUALITY VALIDATION")
print("=" * 50)

# 1. Dataset overview
print("\nDataset shape:", df.shape)

# 2. Missing values
print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 3. Duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# 4. Duplicate property IDs
print("Duplicate IDs:", df["ID"].duplicated().sum())

# 5. Invalid size or price
print(
    "Invalid property sizes:",
    (df["Size_in_SqFt"] <= 0).sum()
)

print(
    "Invalid property prices:",
    (df["Price_in_Lakhs"] <= 0).sum()
)

# 6. Floor consistency
invalid_floors = df["Floor_No"] > df["Total_Floors"]

print(
    "Floor_No greater than Total_Floors:",
    invalid_floors.sum()
)

# 7. Property age consistency
age_mismatch = (
    df["Age_of_Property"]
    != 2025 - df["Year_Built"]
)

print(
    "Property age mismatches against 2025:",
    age_mismatch.sum()
)

# 8. Price per square foot verification
df["Calculated_Price_per_SqFt_Rs"] = (
    df["Price_in_Lakhs"] * 100000
    / df["Size_in_SqFt"]
)

print("\nSupplied price per sq ft summary:")
print(df["Price_per_SqFt"].describe())

print("\nCalculated price per sq ft in rupees:")
print(df["Calculated_Price_per_SqFt_Rs"].describe())

# 9. Category distribution
print("\nOwner type distribution:")
print(df["Owner_Type"].value_counts())

print("\nAvailability status distribution:")
print(df["Availability_Status"].value_counts())

print("\nDATA QUALITY CHECK COMPLETE")


# ==========================================
# INVESTIGATE FLOOR NUMBER INCONSISTENCIES
# ==========================================

print("\n" + "=" * 50)
print("FLOOR NUMBER INVESTIGATION")
print("=" * 50)

invalid_floors = df[
    df["Floor_No"] > df["Total_Floors"]
]

print("\nNumber of inconsistent records:", len(invalid_floors))

print("\nFirst 15 inconsistent records:")
print(
    invalid_floors[
        [
            "City",
            "Property_Type",
            "Floor_No",
            "Total_Floors",
            "Price_in_Lakhs",
            "Size_in_SqFt",
        ]
    ].head(15).to_string(index=False)
)

print("\nFloor number value counts:")
print(df["Floor_No"].value_counts().sort_index())

print("\nTotal floors value counts:")
print(df["Total_Floors"].value_counts().sort_index())

print("\nFloor investigation complete.")


# ==========================================
# PREPARE INITIAL MODELLING DATASET
# ==========================================

from pathlib import Path

print("\n" + "=" * 50)
print("PREPARING MODELLING DATASET")
print("=" * 50)

# Create the processed-data folder
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Start with a copy of the original dataframe
model_df = df.copy()

# Calculate price per square foot in rupees
model_df["Calculated_Price_per_SqFt_Rs"] = (
    model_df["Price_in_Lakhs"] * 100000
    / model_df["Size_in_SqFt"]
)

# Remove unreliable floor-related columns from the initial model
model_df = model_df.drop(
    columns=["Floor_No", "Total_Floors"],
    errors="ignore"
)

# Save the prepared dataset separately
output_path = PROCESSED_DIR / "housing_model_data.csv"
model_df.to_csv(output_path, index=False)

print("\nOriginal dataset shape:", df.shape)
print("Prepared dataset shape:", model_df.shape)
print("Saved to:", output_path)

print("\nPrepared dataset columns:")
print(model_df.columns.tolist())

print("\nInitial modelling dataset prepared successfully!")
