# Bitcoin Price Analysis Before the 2024 Halving

## Introduction
In this project, we analyse Bitcoin prices to forecast trends leading up to the 2024 halving event. The halving event, occurring approximately every four years, historically influences Bitcoin's price. This analysis aims to provide insights into potential price movements and market sentiment before and after this significant event.

## Data Exploration
The dataset used includes historical Bitcoin prices, volumes and market caps. Key statistics and visualisations are provided below to illustrate the data's behavior and trends.

### Summary Statistics
- **Mean Price**: $15,650.33
- **Median Price**: $8,677.05

### Visualizations
![Price Trend](price_trend.png)
*Figure 1: Bitcoin Price Trend Over Time*

## Data Preprocessing
Data preprocessing involves several key steps:
- **Handling Missing Values**: Identifying and addressing any missing data points.
- **Timestamp Conversion**: Converting date columns to datetime format for better analysis.
- **Feature Engineering**: Calculating additional features such as daily returns and rolling averages to enhance model performance.

## Exploratory Data Analysis (EDA)
EDA is crucial for understanding data distributions and trends:
- **Price Trend Analysis**: Line plots show Bitcoin's price movements over time.
- **Price Range Analysis**: Plots indicating high and low prices provide insights into price volatility.
- **Distribution of Daily Returns**: Histograms reveal the distribution and frequency of daily price returns.

## Modeling
We employed the following models for price prediction:
- **Linear Regression**: A fundamental machine learning model for continuous variable prediction.
- **LSTM (Long Short-Term Memory)**: Advanced deep learning model tailored for time series forecasting.

### Model Performance
- **Linear Regression**: RMSE = 2,533.99
- **Linear Regression (Fine-Tuned)**: RMSE = 2,534.08
- **LSTM**: (Add LSTM RMSE and other relevant performance metrics here if available)

## Results
The trained models provide forecasts of Bitcoin prices post-halving. Performance metrics and predictions are compared with actual prices to assess accuracy. 

## Conclusion
The analysis offers valuable insights into Bitcoin price trends around the 2024 halving event. The findings can guide traders and investors in making informed decisions based on historical data and predictive modeling.

## Appendix
Additional code snippets, references and detailed explanations are provided for further understanding and reproducibility of the analysis.
