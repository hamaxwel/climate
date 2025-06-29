# carbon_emissions_regression.py
"""
SDG 13: Climate Action - Predicting CO2 Emissions using Machine Learning
Dataset: owid-co2-data.csv

This script builds a machine learning model to predict annual CO2 emissions for the United States using a Random Forest Regressor and multiple relevant features. The results support SDG 13 by providing actionable insights for climate action.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def load_data(filepath):
    """
    Load the CO2 dataset from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_csv(filepath)


def preprocess_data(df):
    """
    Preprocess the data for modeling:
    - Filter for a specific country (United States)
    - Select relevant features and target
    - Drop rows with missing values
    Args:
        df (pd.DataFrame): Raw data.
    Returns:
        X (pd.DataFrame): Features for modeling.
        y (pd.Series): Target variable (CO2 emissions).
    """
    country = 'United States'
    df_country = df[df['country'] == country]
    # Features selected based on relevance to CO2 emissions
    features = [
        'year', 'gdp', 'population', 'energy_per_capita',
        'primary_energy_consumption', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2'
    ]
    target = 'co2'
    # Drop rows with missing values in selected columns
    data = df_country[features + [target]].dropna()
    X = data[features]
    y = data[target]
    return X, y


def train_model(X_train, y_train):
    """
    Train a Random Forest regression model on the training data.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    Returns:
        model: Trained RandomForestRegressor model.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print key metrics.
    Args:
        model: Trained regression model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    Returns:
        y_pred (np.ndarray): Predicted values for the test set.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.2f}")
    return y_pred


def plot_results(y_test, y_pred):
    """
    Plot actual vs. predicted CO2 emissions for visual evaluation.
    Args:
        y_test (pd.Series): Actual CO2 emissions.
        y_pred (np.ndarray): Predicted CO2 emissions.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.title('Actual vs. Predicted CO2 Emissions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.show()


def main():
    # 1. Load data
    df = load_data('owid-co2-data.csv')
    print('Data loaded. Shape:', df.shape)
    print('Columns:', list(df.columns))

    # 2. Preprocess data
    X, y = preprocess_data(df)
    print('Preprocessing complete. Features shape:', X.shape)

    # 3. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train the Random Forest model
    model = train_model(X_train, y_train)
    print('Model training complete.')

    # 5. Evaluate the model and print metrics
    y_pred = evaluate_model(model, X_test, y_test)

    # 6. Visualize the results
    plot_results(y_test, y_pred)


if __name__ == "__main__":
    main() 