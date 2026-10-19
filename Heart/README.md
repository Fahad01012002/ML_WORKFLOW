# Heart Disease Prediction: EDA, Preprocessing & Feature Selection

A Jupyter Notebook that explores the **Heart Failure Prediction** dataset, cleans invalid values, encodes and scales the features, and uses statistical tests to select the features most strongly associated with heart disease.

## Project Overview

The goal is to understand which clinical measurements and patient characteristics are most associated with the presence of heart disease, and to prepare a clean, model-ready dataset for a future classification model.

## Dataset

- **File:** `heart.csv`
- **Size:** 918 rows × 12 columns
- **Missing values:** none (but see the note on zero values below)
- **Duplicates:** none
- **Target:** `HeartDisease` (1 = heart disease, 0 = normal)

| Column | Description |
|--------|-------------|
| `Age` | Age of the patient (years) |
| `Sex` | M / F |
| `ChestPainType` | TA (typical angina), ATA (atypical angina), NAP (non-anginal pain), ASY (asymptomatic) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dl) |
| `FastingBS` | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `RestingECG` | Resting ECG result: Normal, ST (ST-T wave abnormality), LVH (left ventricular hypertrophy) |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression induced by exercise relative to rest |
| `ST_Slope` | Slope of the peak exercise ST segment: Up, Flat, Down |
| `HeartDisease` | Target variable |

> The dataset is not included in this repo by default. Place `heart.csv` in the same folder as the notebook (it is available on Kaggle as "Heart Failure Prediction Dataset").

## Workflow

### 1. Exploratory Data Analysis
- Data types, summary statistics, missing-value and duplicate checks
- Histograms with KDE for `Age`, `RestingBP`, `Cholesterol`, and `MaxHR`
- Count plots of `Sex` and `ChestPainType` split by heart disease status
- Box plots of numeric features against the target
- Correlation heatmap

### 2. Data Cleaning & Preprocessing
- Found **172 rows with `Cholesterol = 0`** and some with `RestingBP = 0`, which are physiologically impossible and treated as missing data
- Replaced these zeros with the mean of the valid (non-zero) values
- One-hot encoded all categorical columns
- Standardized `Age`, `Cholesterol`, `RestingBP`, and `MaxHR` with `StandardScaler`

### 3. Feature Selection
- **Pearson correlation** of every feature with `HeartDisease`
- **Chi-square test of independence** (α = 0.05) between each feature and the target
- Dropped features that showed no significant relationship: `ChestPainType_TA` and `RestingECG_LVH`

## Key Findings

- **`ST_Slope_Flat`** has the strongest positive correlation with heart disease (r ≈ 0.55), followed by **asymptomatic chest pain** (`ChestPainType_ASY`, r ≈ 0.52) and **exercise-induced angina** (r ≈ 0.49).
- **`Oldpeak`** (r ≈ 0.39), **male sex** (r ≈ 0.31), **age** (r ≈ 0.28), and **high fasting blood sugar** (r ≈ 0.27) are also positively associated.
- `ST_Slope_Up` shows the strongest chi-square association with the target.
- `Cholesterol` and `RestingBP` show weak linear correlation with the target.

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
3. Add `heart.csv` to the project folder
4. Launch the notebook
   ```bash
   jupyter notebook Heart.ipynb
   ```

## Repository Structure

```
├── Heart.ipynb    # Main analysis notebook
├── heart.csv      # Dataset (add this yourself)
└── README.md
```

## Next Steps

- Split into train/test sets and train classifiers (Logistic Regression, Random Forest, XGBoost)
- Evaluate with accuracy, precision, recall, F1, and ROC-AUC, with extra attention to recall since missing a heart disease case is costly
- Keep `Oldpeak` as a decimal rather than casting to integer, to preserve information
- Fit the scaler and imputation values on the training set only, to avoid data leakage
- Try median or model-based imputation for the zero-valued `Cholesterol` rows

## License

This project is open source and available under the [MIT License](LICENSE).
