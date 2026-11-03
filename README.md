# 🏡 Real Estate Investment Advisor
### Property Analytics, Machine Learning Evaluation & Future Value Scenarios

An interactive data analytics application built with Python and Streamlit to explore Indian housing data, compare property prices, filter properties based on user preferences, and calculate hypothetical future property values.

## 📌 Project Overview

The Real Estate Investment Advisor helps users explore property listings through interactive visualizations and data-driven analysis.

The application provides property market summaries, city and property-type comparisons, property filtering, and a future-value calculator based on user-selected appreciation assumptions.

The project also evaluates whether the available housing dataset supports reliable machine-learning-based property price estimation.

## 🎯 Objectives

- Explore housing prices across Indian states and cities.
- Analyze property prices, sizes, and property types.
- Build an interactive real estate analytics dashboard.
- Train and evaluate a machine learning regression model.
- Compare model performance against a simple baseline.
- Calculate hypothetical future property values under different appreciation assumptions.
- Communicate dataset limitations transparently.

## ✨ Features

### 📊 Property Market Overview
- Total properties analyzed.
- Median listed property price.
- Median property size.
- Median calculated price per square foot.

### 🔎 Interactive Property Filters
- Filter by state.
- Filter by city.
- Filter by property type.
- Filter by price range.

### 📈 Data Visualization
- Property price distribution.
- Median price comparison by property type.
- Cities with the highest median listed prices.
- Property size versus listed price.

### 💡 Investment Analysis
- Filter properties using a budget.
- Select a preferred number of bedrooms.
- Browse properties matching the selected criteria.
- Compare listed prices and property sizes.

### 🔮 Future Value Scenario Calculator
- Enter a starting property price.
- Choose an assumed annual appreciation rate.
- Select an investment period.
- View the resulting hypothetical value and percentage change.

### 📥 Property Data Export
Download filtered property records as a CSV file.

## 🧰 Technology Stack

- **Language:** Python
- **Data Processing:** Pandas
- **Visualization:** Plotly
- **Dashboard:** Streamlit
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
RealEstateInvestmentAdvisor/
│
├── data/
│   ├── india_housing_prices.csv
│   └── processed/
│       └── housing_model_data.csv
│
├── models/
│   └── property_price_model.joblib
│
├── outputs/
│   ├── charts/
│   ├── correlation_matrix.csv
│   ├── data_audit.csv
│   └── numeric_summary.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── predict.py
│   └── train_model.py
│
├── app.py
├── README.md
└── requirements.txt
```

## 📁 Dataset

The project uses an Indian housing dataset containing **250,000 property records** and **23 original columns**.

The dataset includes information such as:

- State, city, and locality.
- Property type and number of bedrooms.
- Property size in square feet.
- Listed price in lakhs.
- Year built and property age.
- Nearby schools and hospitals.
- Furnishing status, parking, and security.
- Amenities and availability status.

The original dataset was checked for missing values, duplicate rows, duplicate IDs, and potential data-quality issues.

The preprocessing pipeline creates a processed dataset for analysis and model training while retaining the original CSV file.

## 🤖 Machine Learning Evaluation

A Random Forest regression model was trained to estimate `Price_in_Lakhs`.

The evaluation used an 80/20 training and testing split.

| Metric | Result |
|---|---:|
| Training records | 200,000 |
| Testing records | 50,000 |
| Mean Absolute Error (MAE) | ₹122.32 lakhs |
| Root Mean Squared Error (RMSE) | ₹141.22 lakhs |
| R² score | -0.0004 |

A mean-prediction baseline was also evaluated.

The Random Forest model did not meaningfully outperform the baseline. Its predictions were concentrated near the average price instead of capturing the variation in actual property prices.

### Interpretation

These results indicate that the available dataset, as currently prepared, does not support reliable property price estimation using the trained model.

The model is retained as an evaluation result rather than presented as a reliable property valuation tool.

## ⚠️ Important Limitations

1. The dataset does not contain verified future property prices or actual five-year appreciation labels.
2. The dataset does not establish a reliable supervised-learning target for determining whether a property is a good investment.
3. The trained model did not outperform a simple mean-prediction baseline.
4. The future-value calculator uses user-selected assumptions, not machine-learning forecasts.
5. The investment screening section filters listings according to budget and bedroom preferences; it does not establish investment quality.
6. The dataset's listed prices have not been independently verified as current market prices.
7. The dataset does not provide sufficient verified rental income, ownership costs, or transaction expenses to calculate a complete investment return.

Consequently, this application is an educational property analytics tool and should not be treated as professional financial or real estate advice.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your computer.
- Visual Studio Code or another Python editor.
- The project dataset in the expected `data/` directory.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Open the project directory

```bash
cd RealEstateInvestmentAdvisor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

If the requirements file is not yet complete, install the dashboard dependencies with:

```bash
pip install streamlit pandas plotly
```

### 6. Run the dashboard

```bash
streamlit run app.py
```

Streamlit will provide a local address where you can access the application in your browser.

## 🚀 Future Improvements

- Find a reliable, geographically consistent housing dataset.
- Investigate meaningful price predictors and data quality.
- Evaluate models using cross-validation and appropriate baselines.
- Incorporate verified historical prices if available.
- Add rental yield and ownership-cost analysis when suitable data becomes available.
- Improve the dashboard with additional comparisons and reporting.
- Deploy the application online.

## 👩‍💻 Author

**Arpita Behera**

Developed as a real estate analytics and machine learning project using Python.

---

*This project is intended for educational and analytical purposes. All future-value calculations are illustrative scenarios, not guaranteed forecasts.*