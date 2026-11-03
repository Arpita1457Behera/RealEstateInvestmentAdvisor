from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DASHBOARD THEME
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7F4;
        color: #1E3328;
    }

    [data-testid="stSidebar"] {
        background-color: #173D2B;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    h1 {
        color: #173D2B;
        font-weight: 800;
    }

    h2, h3 {
        color: #24563D;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE7DD;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(23, 61, 43, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #52685A;
    }

    [data-testid="stMetricValue"] {
        color: #173D2B;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background-color: #24563D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #367451;
        color: white;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #DCE7DD;
    }

    /* Make warning messages readable */
    [data-testid="stAlert"] {
        background-color: #FFF8DB;
        color: #493B0B;
        border: 1px solid #E8D58A;
        border-radius: 12px;
    }

    [data-testid="stAlert"] p {
        color: #493B0B !important;
    }

    /* Improve KPI card spacing */
    [data-testid="stMetric"] {
        min-height: 125px;
        padding: 18px 14px;
        overflow-wrap: anywhere;
    }

    [data-testid="stMetricValue"] {
        font-size: clamp(1.25rem, 1.7vw, 2rem);
        white-space: normal;
        overflow-wrap: anywhere;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
    }

</style>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DATA = BASE_DIR / "data" / "processed" / "housing_model_data.csv"
RAW_DATA = BASE_DIR / "data" / "india_housing_prices.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    if PROCESSED_DATA.exists():
        data = pd.read_csv(PROCESSED_DATA)
    elif RAW_DATA.exists():
        data = pd.read_csv(RAW_DATA)
    else:
        raise FileNotFoundError(
            "Dataset not found. Check the files inside the data folder."
        )

    numeric_columns = [
        "Price_in_Lakhs",
        "Size_in_SqFt",
        "BHK",
        "Year_Built",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Calculated_Price_per_SqFt_Rs"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    if "Calculated_Price_per_SqFt_Rs" not in data.columns:
        if {"Price_in_Lakhs", "Size_in_SqFt"}.issubset(data.columns):
            data["Calculated_Price_per_SqFt_Rs"] = (
                data["Price_in_Lakhs"] * 100000
                / data["Size_in_SqFt"].replace(0, float("nan"))
            )

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏡 Real Estate Investment Advisor")
st.subheader("Property Analytics & Future Value Scenarios")

st.markdown(
    """
    Explore housing data across Indian cities, compare property prices,
    and calculate hypothetical future-value scenarios.
    """
)

st.warning(
    "Important limitation: The current dataset does not establish reliable "
    "relationships between property features and prices. The trained model "
    "did not outperform a simple average-price baseline. This dashboard "
    "provides descriptive analytics and hypothetical scenarios—not reliable "
    "price predictions or guaranteed investment advice."
)

st.divider()


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("🔎 Explore Properties")
st.sidebar.caption("Use these filters to narrow down the dataset.")

filtered_df = df.copy()


def add_filter(data, column, label):
    if column not in data.columns:
        return data

    options = sorted(
        data[column].dropna().astype(str).unique().tolist()
    )

    selected = st.sidebar.multiselect(
        label,
        options,
        default=[]
    )

    if selected:
        data = data[data[column].astype(str).isin(selected)]

    return data


filtered_df = add_filter(filtered_df, "State", "Select State")
filtered_df = add_filter(filtered_df, "City", "Select City")
filtered_df = add_filter(
    filtered_df, "Property_Type", "Property Type"
)

if "Price_in_Lakhs" in filtered_df.columns:
    valid_prices = filtered_df["Price_in_Lakhs"].dropna()

    if not valid_prices.empty:
        minimum_price = float(valid_prices.min())
        maximum_price = float(valid_prices.max())

        if minimum_price < maximum_price:
            selected_range = st.sidebar.slider(
                "Price Range (₹ Lakhs)",
                min_value=minimum_price,
                max_value=maximum_price,
                value=(minimum_price, maximum_price)
            )

            filtered_df = filtered_df[
                filtered_df["Price_in_Lakhs"].between(
                    selected_range[0], selected_range[1]
                )
            ]


if filtered_df.empty:
    st.warning("No properties match your filters. Please change them.")
    st.stop()


# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------

st.header("📊 Property Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Properties Analysed",
    f"{len(filtered_df):,}"
)

if "Price_in_Lakhs" in filtered_df.columns:
    median_price = filtered_df["Price_in_Lakhs"].median()

    col2.metric(
        "Median Property Price",
        f"₹{median_price:,.1f}L"
        if pd.notna(median_price) else "N/A"
    )

if "Size_in_SqFt" in filtered_df.columns:
    median_size = filtered_df["Size_in_SqFt"].median()

    col3.metric(
        "Median Property Size",
        f"{median_size:,.0f} sq ft"
        if pd.notna(median_size) else "N/A"
    )

if "Calculated_Price_per_SqFt_Rs" in filtered_df.columns:
    median_rate = filtered_df["Calculated_Price_per_SqFt_Rs"].median()

    col4.metric(
        "Median Calculated Rate",
        f"₹{median_rate:,.0f}/sq ft"
        if pd.notna(median_rate) else "N/A"
    )

st.caption(
    "These figures describe the available dataset; they are not verified "
    "current market rates. Calculated rates are derived from the listed "
    "price and property size."
)

st.divider()


# --------------------------------------------------
# PRICE DISTRIBUTION
# --------------------------------------------------

st.header("💰 Property Price Analysis")

if "Price_in_Lakhs" in filtered_df.columns:
    price_data = filtered_df.dropna(subset=["Price_in_Lakhs"])

    if not price_data.empty:
        fig_price = px.histogram(
            price_data,
            x="Price_in_Lakhs",
            nbins=40,
            title="Distribution of Property Prices",
            labels={
                "Price_in_Lakhs": "Property Price (₹ Lakhs)",
                "count": "Number of Properties"
            }
        )

        fig_price.update_layout(
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig_price, use_container_width=True)


# --------------------------------------------------
# PROPERTY TYPE AND CITY COMPARISON
# --------------------------------------------------

left_chart, right_chart = st.columns(2)

with left_chart:
    if {"Property_Type", "Price_in_Lakhs"}.issubset(filtered_df.columns):
        type_prices = (
            filtered_df.dropna(
                subset=["Property_Type", "Price_in_Lakhs"]
            )
            .groupby("Property_Type", as_index=False)
            ["Price_in_Lakhs"]
            .median()
            .sort_values("Price_in_Lakhs", ascending=False)
        )

        fig_type = px.bar(
            type_prices,
            x="Property_Type",
            y="Price_in_Lakhs",
            title="Median Price by Property Type",
            labels={
                "Property_Type": "Property Type",
                "Price_in_Lakhs": "Median Price (₹ Lakhs)"
            },
            text_auto=".2f"
        )

        fig_type.update_layout(
            template="plotly_white",
            height=420
        )

        st.plotly_chart(fig_type, use_container_width=True)

with right_chart:
    if {"City", "Price_in_Lakhs"}.issubset(filtered_df.columns):
        city_prices = (
            filtered_df.dropna(subset=["City", "Price_in_Lakhs"])
            .groupby("City", as_index=False)
            .agg(
                Median_Price=("Price_in_Lakhs", "median"),
                Property_Count=("Price_in_Lakhs", "count")
            )
            .sort_values("Median_Price", ascending=False)
            .head(10)
        )

        fig_city = px.bar(
            city_prices.sort_values("Median_Price"),
            x="Median_Price",
            y="City",
            orientation="h",
            title="10 Cities with Highest Median Prices",
            labels={
                "Median_Price": "Median Price (₹ Lakhs)",
                "City": "City"
            },
            hover_data=["Property_Count"]
        )

        fig_city.update_layout(
            template="plotly_white",
            height=420
        )

        st.plotly_chart(fig_city, use_container_width=True)

st.divider()


# --------------------------------------------------
# PROPERTY SIZE VS PRICE
# --------------------------------------------------

st.header("📐 Property Size vs Price")

if {"Size_in_SqFt", "Price_in_Lakhs"}.issubset(filtered_df.columns):
    scatter_data = filtered_df.dropna(
        subset=["Size_in_SqFt", "Price_in_Lakhs"]
    )

    # Limit chart points for a smoother experience.
    if len(scatter_data) > 5000:
        scatter_data = scatter_data.sample(
            n=5000, random_state=42
        )

    if not scatter_data.empty:
        color_column = (
            "Property_Type"
            if "Property_Type" in scatter_data.columns
            else None
        )

        fig_scatter = px.scatter(
            scatter_data,
            x="Size_in_SqFt",
            y="Price_in_Lakhs",
            color=color_column,
            title="Property Size and Listed Price",
            labels={
                "Size_in_SqFt": "Size (sq ft)",
                "Price_in_Lakhs": "Listed Price (₹ Lakhs)"
            },
            opacity=0.65
        )

        fig_scatter.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.caption(
            "Up to 5,000 records are sampled for this chart when the "
            "filtered dataset is larger. A visual association does not "
            "prove that size causes a particular price."
        )

st.divider()


# --------------------------------------------------
# FUTURE VALUE SCENARIO CALCULATOR
# --------------------------------------------------

st.header("🔮 Hypothetical Future Value Calculator")

st.write(
    "Choose an assumed annual appreciation rate to explore how a property's "
    "value might change over time. The rate is your assumption, not a "
    "forecast produced by the dataset or the machine-learning model."
)

calc_col1, calc_col2, calc_col3 = st.columns(3)

with calc_col1:
    current_price = st.number_input(
        "Current Property Price (₹ Lakhs)",
        min_value=0.01,
        value=50.0,
        step=5.0
    )

with calc_col2:
    annual_rate = st.number_input(
        "Assumed Annual Appreciation (%)",
        min_value=-50.0,
        max_value=100.0,
        value=5.0,
        step=0.5
    )

with calc_col3:
    years = st.slider(
        "Investment Period (Years)",
        min_value=1,
        max_value=20,
        value=5
    )

future_price = current_price * (
    1 + annual_rate / 100
) ** years

estimated_gain = future_price - current_price

result1, result2, result3 = st.columns(3)

result1.metric(
    "Starting Price",
    f"₹{current_price:,.2f} Lakhs"
)

result2.metric(
    f"Hypothetical Value After {years} Years",
    f"₹{future_price:,.2f} Lakhs",
    delta=f"₹{estimated_gain:,.2f} Lakhs"
)

result3.metric(
    "Hypothetical Percentage Change",
    f"{((future_price / current_price) - 1) * 100:,.2f}%"
)

scenario_years = list(range(years + 1))
scenario_values = [
    current_price * (1 + annual_rate / 100) ** year
    for year in scenario_years
]

scenario_df = pd.DataFrame({
    "Year": scenario_years,
    "Hypothetical Property Value (₹ Lakhs)": scenario_values
})

fig_scenario = px.line(
    scenario_df,
    x="Year",
    y="Hypothetical Property Value (₹ Lakhs)",
    markers=True,
    title="Illustrative Property Value Scenario"
)

fig_scenario.update_layout(
    template="plotly_white",
    height=400
)

st.plotly_chart(fig_scenario, use_container_width=True)

st.info(
    "This calculator assumes the same annual appreciation rate every year. "
    "It excludes taxes, registration fees, maintenance, financing costs, "
    "rental income, inflation and changes in market conditions. Actual "
    "property values can rise or fall."
)

st.divider()


# --------------------------------------------------
# PROPERTY EXPLORER
# --------------------------------------------------

st.header("🏘️ Explore Individual Properties")

available_columns = [
    column for column in [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Year_Built",
        "Furnished_Status",
        "Parking_Space",
        "Security",
        "Availability_Status"
    ]
    if column in filtered_df.columns
]

if available_columns:
    st.dataframe(
        filtered_df[available_columns].head(500),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing up to 500 matching records. Use the sidebar filters "
        "to narrow the results."
    )

    csv_data = filtered_df[available_columns].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Properties as CSV",
        data=csv_data,
        file_name="filtered_properties.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# INVESTMENT ANALYSIS
# --------------------------------------------------

st.header("💡 Investment Analysis")

st.write(
    "Evaluate properties using your own budget and preferences. "
    "This is a preliminary screening tool, not a guaranteed investment "
    "recommendation."
)

analysis_col1, analysis_col2 = st.columns(2)

with analysis_col1:
    investment_budget = st.number_input(
        "Your Maximum Budget (₹ Lakhs)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
        key="investment_budget"
    )

with analysis_col2:
    preferred_bhk = st.selectbox(
        "Preferred Number of Bedrooms",
        ["Any", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5+ BHK"],
        key="preferred_bhk"
    )

# Apply the user's screening criteria.
investment_df = filtered_df.copy()

if "Price_in_Lakhs" in investment_df.columns:
    investment_df = investment_df[
        investment_df["Price_in_Lakhs"] <= investment_budget
    ]

if preferred_bhk != "Any" and "BHK" in investment_df.columns:
    if preferred_bhk == "5+ BHK":
        investment_df = investment_df[
            investment_df["BHK"] >= 5
        ]
    else:
        bhk_number = int(preferred_bhk.split()[0])
        investment_df = investment_df[
            investment_df["BHK"] == bhk_number
        ]

if investment_df.empty:
    st.warning(
        "No properties match your budget and bedroom preferences. "
        "Try increasing your budget or changing your filters."
    )
else:
    st.success(
        f"Found {len(investment_df):,} properties matching your criteria."
    )

    display_columns = [
        column for column in [
            "State",
            "City",
            "Locality",
            "Property_Type",
            "BHK",
            "Size_in_SqFt",
            "Price_in_Lakhs",
            "Calculated_Price_per_SqFt_Rs"
        ]
        if column in investment_df.columns
    ]

    st.dataframe(
        investment_df[display_columns]
        .sort_values("Price_in_Lakhs")
        .head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Properties are ranked by listed price, not investment quality. "
        "The dataset does not establish rental yield, future appreciation, "
        "ownership costs, or verified current market value."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Real Estate Investment Advisor | Built with Python, Pandas, "
    "Plotly and Streamlit | Educational project"
)