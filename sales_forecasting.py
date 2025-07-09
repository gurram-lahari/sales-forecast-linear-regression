import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data (using dummy data for demonstration)
data = {
    'date': pd.date_range(start='2023-01-01', periods=24, freq='M'),
    'product': ['A'] * 24,
    'quantity': np.random.randint(50, 200, 24),
    'revenue': np.random.randint(1000, 5000, 24)
}
df = pd.DataFrame(data)

# 2. Preprocess Data
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
# Handle nulls (if any)
df = df.fillna(method='ffill')
# Feature engineering: extract month and year
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
# Use month as a time index for regression
# (for real data, you may want to use more features)
df['time_index'] = np.arange(len(df))

# 3. Prepare Features and Target
X = df[['time_index']]
y = df['revenue']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))

# 8. Plot Actual vs Predicted
plt.figure(figsize=(10,6))
plt.plot(df['date'], y, label='Actual Revenue')
plt.plot(df['date'].iloc[len(X_train):], y_pred, label='Predicted Revenue', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Actual vs Predicted Sales Revenue')
plt.legend()
plt.tight_layout()
plt.show()

# 9. Tabular Forecast
forecast_df = df.iloc[len(X_train):][['date', 'revenue']].copy()
forecast_df['predicted_revenue'] = y_pred
print('\nTabular Forecast:')
print(forecast_df)

# 10. Forecast for Next 3 Months
future_time_idx = np.arange(len(df), len(df)+3)
future_pred = model.predict(future_time_idx.reshape(-1,1))
future_dates = pd.date_range(start=df['date'].iloc[-1]+pd.offsets.MonthEnd(1), periods=3, freq='M')
future_forecast = pd.DataFrame({'date': future_dates, 'predicted_revenue': future_pred})
print('\nFuture 3-Month Forecast:')
print(future_forecast) 