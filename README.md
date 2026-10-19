# Medical Insurance Charges: EDA, Preprocessing & Feature Selection

A Jupyter Notebook that explores the well-known **Medical Cost Personal** insurance dataset, cleans and encodes it, engineers new features, and uses statistical tests to select the features that matter most for predicting individual medical insurance charges.

## Project Overview

The goal is to understand which personal factors (age, smoking status, BMI, etc.) drive medical insurance costs, and to prepare a clean, model-ready dataset for a future regression model.

## Dataset

- **File:** `insurance.csv`
- **Size:** 1,338 rows × 7 columns
- **Missing values:** none

| Column | Description |
|--------|-------------|
| `age` | Age of the primary beneficiary |
| `sex` | Gender (male / female) |
| `bmi` | Body mass index |
| `children` | Number of dependents covered |
| `smoker` | Whether the person smokes (yes / no) |
| `region` | Residential region in the US (northeast, northwest, southeast, southwest) |
| `charges` | Individual medical costs billed by insurance (**target variable**) |

> The dataset is not included in this repo by default. Place `insurance.csv` in the same folder as the notebook (it is available on Kaggle as "Medical Cost Personal Datasets").

## Workflow

### 1. Exploratory Data Analysis
- Summary statistics, data types, and missing-value checks
- Histograms with KDE for `age`, `bmi`, `children`, and `charges`
- Count plots for `sex`, `region`, `children`, and `smoker`
- Box plots to inspect outliers
- Correlation heatmap of numeric features

### 2. Data Cleaning & Preprocessing
- Removed 1 duplicate row (1,338 → 1,337)
- Encoded `sex` → `is_female` and `smoker` → `is_smoker` (binary)
- One-hot encoded `region`
- Converted all columns to integers

### 3. Feature Engineering & Selection
- Created a `bmi_category` feature (Underweight, Normal, Overweight, Obese) and one-hot encoded it
- Standardized `age`, `bmi`, and `children` using `StandardScaler`
- **Pearson correlation** of each feature with `charges`
- **Chi-square test of independence** (α = 0.05) against quartile-binned charges
- Selected final features: `age`, `is_female`, `bmi`, `children`, `is_smoker`, `region_southeast`, `bmi_category_Obese` (plus the `charges` target)

## Key Findings

- **Smoking is by far the strongest predictor of charges** (Pearson r ≈ 0.79).
- **Age** is the second strongest (r ≈ 0.30), followed by obesity (r ≈ 0.20) and BMI (r ≈ 0.20).
- Number of children and most region indicators have weak linear correlation with charges.
- Sex and region are balanced in the data; about 20% of customers are smokers.

## Tech Stack

- Python 3
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (`StandardScaler`)
- SciPy (`pearsonr`, `chi2_contingency`)

## Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Install dependencies
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn scipy jupyter
   ```
3. Add `insurance.csv` to the project folder
4. Launch the notebook
   ```bash
   jupyter notebook Insurance.ipynb
   ```

## Repository Structure

```
├── Insurance.ipynb    # Main analysis notebook
├── insurance.csv      # Dataset (add this yourself)
└── README.md
```

## Next Steps

- Train and compare regression models (Linear Regression, Random Forest, Gradient Boosting)
- Evaluate with R², MAE, and RMSE using a train/test split and cross-validation
- Try a log transform on `charges` to handle its right-skewed distribution
- Explore interaction effects such as smoker × BMI

## License

This project is open source and available under the [MIT License](LICENSE).
