# Week 1 Task - Titanic Data Cleaning & Exploratory Analysis

This repo is my Week 1 submission for the Data Science internship. The task was to
practice the full data preparation workflow that comes before any actual modeling:
finding a public dataset, cleaning it up, and exploring it with summary statistics
and visualizations.

## Project Description

This project covers the complete data preparation workflow: data acquisition,
cleaning, and exploratory data analysis (EDA) using Python. The goal was to
simulate the initial phase of a real data science project, where raw data is
rarely ready for analysis and must first be inspected, cleaned, and understood
before any modeling can happen.

For this task, I used the Titanic passenger dataset, a well-known public dataset
containing 891 records with 12 columns describing passenger details such as age,
sex, ticket class, fare paid, and survival outcome. I chose this dataset because
it is small enough to review manually column by column, while still containing
realistic data quality issues like missing values, which made it a good fit for
practicing cleaning techniques.

Using Pandas, I checked for missing values, duplicate rows, and incorrect data
types. I dropped the `Cabin` column due to excessive missing data (77%), filled
missing `Age` values with the median, and filled missing `Embarked` values with
the mode, documenting the reasoning behind each decision. I also engineered a
new `FamilySize` feature from existing columns.

For the exploratory analysis, I used Matplotlib to create five visualizations: a
missing-values bar chart, an age distribution histogram, a survival-rate-by-class
bar chart, a fare boxplot comparing survivors and non-survivors, and a
correlation heatmap of numeric features. Key insights include the strong
influence of sex and passenger class on survival rate.

## Dataset

- **Source:** [datasciencedojo/datasets](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) (public GitHub mirror of the classic Kaggle Titanic dataset)
- **Size:** 891 rows, 12 columns
- **Target-like column:** `Survived` (0 = died, 1 = survived)

## Repo Structure

```
├── data/
│   ├── titanic_raw.csv        # original, unmodified dataset
│   └── titanic_clean.csv      # cleaned dataset after processing
├── notebook/
│   └── titanic_eda.ipynb      # full code: cleaning + EDA + charts
├── images/
│   ├── 1_missing_values.png
│   ├── 2_age_histogram.png
│   ├── 3_survival_by_class.png
│   ├── 4_fare_boxplot.png
│   └── 5_correlation_heatmap.png
├── report/
│   └── Week1_Titanic_Data_Report.docx   # full written report
└── README.md
```

## How to Run

1. Clone this repo
2. Install requirements: `pip install pandas matplotlib`
3. Open `notebook/titanic_eda.ipynb` in Jupyter and run all cells

## Steps Covered

1. **Data Acquisition** - loaded the CSV with pandas
2. **Data Cleaning**
   - Checked for missing values, duplicates, and wrong data types
   - Dropped `Cabin` (77% missing)
   - Filled `Age` with the median
   - Filled `Embarked` with the mode
   - Converted categorical columns to the `category` dtype
   - Engineered a `FamilySize` feature
3. **Exploratory Data Analysis**
   - Summary statistics
   - 5 visualizations (missing values, age histogram, survival by class, fare boxplot, correlation heatmap)
4. **Insights** - documented in the notebook and the full Word report

## Key Insights

- Sex was the strongest predictor of survival: ~74% of female passengers survived vs. ~19% of male passengers.
- Passenger class mattered a lot: 63% (1st class) vs 47% (2nd class) vs 24% (3rd class) survival rate.
- Fare and Pclass are strongly (negatively) correlated with each other, and Fare has a modest positive correlation with survival.
- Family size had a non-linear relationship with survival - small families (2-4 people) survived at higher rates than solo travelers or very large families.
- The Cabin column was too incomplete (77% missing) to be usable, so it was dropped.

## Full Report

See `report/Week1_Titanic_Data_Report.docx` for the complete write-up with
methodology, rationale, code, charts, and insights.
