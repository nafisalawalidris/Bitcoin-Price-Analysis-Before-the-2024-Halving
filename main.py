import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(file_path):
    """Load Bitcoin price data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data: handle missing values, convert dates, calculate features."""
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate additional features: Daily Returns and Rolling Averages
    df['Daily_Returns'] = df['Close'].pct_change()
    df['Rolling_Average'] = df['Close'].rolling(window=30).mean()

    # Drop rows with missing values
    df.dropna(inplace=True)
    return df

def perform_eda(df):
    """Perform Exploratory Data Analysis (EDA) on the data."""
    # Line plot to visualise Bitcoin price over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], color='blue')
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()

    # Plot Bitcoin price over time with high and low prices indicated
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], color='blue', label='Close')
    plt.plot(df['Date'], df['High'], color='green', linestyle='--', label='High')
    plt.plot(df['Date'], df['Low'], color='red', linestyle='--', label='Low')
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Histogram to analyse the distribution of daily returns
    plt.figure(figsize=(10, 6))
    plt.hist(df['Daily_Returns'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def calculate_summary_statistics(df):
    """Calculate and print summary statistics."""
    mean_price = df['Close'].mean()
    median_price = df['Close'].median()
    print(f"### Summary Statistics\n- **Mean Price**: ${mean_price:.2f}\n- **Median Price**: ${median_price:.2f}")

def train_model(df):
    """Train a Linear Regression model on the data."""
    # Define features and target variable
    X = df[['Daily_Returns', 'Rolling_Average']]
    y = df['Close']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"### Model Performance\n- **Linear Regression**: RMSE = {rmse:.2f}")

    return model, X, y, X_train, X_test, y_train, y_test

def predict_prices(model, X):
    """Use the trained model to make predictions on the data."""
    predicted_prices = model.predict(X)
    predicted_prices_df = pd.DataFrame({'Predicted_Price': predicted_prices})
    print("Predicted Prices:\n", predicted_prices_df)

def fine_tune_model(X_train, y_train, X_test, y_test):
    """Fine-tune the Linear Regression model using Grid Search."""
    # Define the parameter grid to search
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(estimator=LinearRegression(), param_grid=param_grid, cv=5)

    # Perform grid search to find the best parameters
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Use the best parameters to initialize the model
    best_model = LinearRegression(**best_params)
    best_model.fit(X_train, y_train)

    # Evaluate the fine-tuned model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"### Model Performance\n- **Linear Regression (Fine-Tuned)**: RMSE = {rmse:.2f}")

    return best_model

def main():
    # File path to the CSV containing Bitcoin price data
    file_path = r"C:\Users\USER\Downloads\BTC-USD Yahoo Finance - Max Yrs.csv"

    # Load and preprocess the data
    df = load_data(file_path)
    df = preprocess_data(df)

    # Calculate summary statistics
    calculate_summary_statistics(df)

    # Perform Exploratory Data Analysis (EDA)
    perform_eda(df)

    # Train the model
    model, X, y, X_train, X_test, y_train, y_test = train_model(df)

    # Predict prices
    predict_prices(model, X)

    # Fine-tune the model
    best_model = fine_tune_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
