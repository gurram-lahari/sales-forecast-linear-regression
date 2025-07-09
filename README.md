# Sales Forecasting with Linear Regression

This project demonstrates how to forecast sales using linear regression in Python. It includes data preprocessing, model training, evaluation, and visualization.

## Features
- Loads sales data (dummy data by default, or replace with your own CSV)
- Handles missing values
- Trains a linear regression model
- Plots actual vs. predicted sales
- Provides tabular and graphical forecasts

## Requirements
- Python 3.7+
- See `requirements.txt` for dependencies

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python sales_forecasting.py
   ```

3. (Optional) To use your own data, replace the dummy data section in `sales_forecasting.py` with:
   ```python
   df = pd.read_csv('your_sales_data.csv')
   ```
   Ensure your CSV has columns: `date`, `product`, `quantity`, `revenue`.

## Output
- Console: Model evaluation metrics and tabular forecasts
- Plots: Actual vs. predicted sales, future forecasts

---

**Author:** Gurram Lahari