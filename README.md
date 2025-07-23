# House-price-Prediction-ML

This repository contains a machine learning project focused on predicting house prices.

## Project Overview

The project demonstrates a complete machine learning workflow, from data exploration and preprocessing to model training and evaluation.

## Key Features

-   **Data Loading & EDA:** Initial data inspection, correlation heatmaps, and distribution analysis of categorical features.
-   **Data Cleaning:** Handling missing values and dropping irrelevant columns.
-   **Feature Engineering:** One-Hot Encoding for categorical variables.
-   **Model Training:** Implementation and comparison of various regression models, including:
    * Support Vector Regressor (SVR)
    * Random Forest Regressor
    * Linear Regression
    * CatBoostRegressor (intended)
-   **Model Evaluation:** Performance assessment using MAE, RMSE, and R² Score.
-   **Visualization:** Plots to compare actual vs. predicted prices.
-   **Model Persistence:** Saving trained models for future use.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Joblib
-   CatBoost (intended)

## How to Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd your-repository-name
    ```
3.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib catboost
    ```
    (Note: `catboost` might require additional installation steps if you encounter issues.)
4.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook HousePricePrediction.ipynb
    ```
5.  Run all cells in the notebook.

## Dataset

The project uses the `HousePricePrediction.xlsx` dataset, which is expected to be in the same directory as the notebook.

## Results

The notebook provides evaluation metrics and visualizations for each trained model, showcasing their performance in predicting house prices.
