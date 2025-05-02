from flask import Flask, jsonify, render_template
from flask_cors import CORS
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

# === FIREBASE SETUP ===
firebase_key_json = os.environ.get("FIREBASE_KEY_JSON")

if not firebase_key_json:
    raise ValueError("FIREBASE_KEY_JSON env var is not set")

cred = credentials.Certificate(json.loads(firebase_key_json))
firebase_admin.initialize_app(cred)
db = firestore.client()

# === HELPER FUNCTIONS ===
def get_sales_data():
    sales_ref = db.collection("sales_orders").order_by("date")
    docs = sales_ref.stream()
    data = []

    for doc in docs:
        entry = doc.to_dict()
        if 'date' in entry and 'total_php' in entry and 'quantity' in entry and 'category' in entry:
            data.append({
                'date': entry['date'],
                'total_php': entry['total_php'],
                'quantity': entry['quantity'],
                'category': entry['category']
            })

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')
    df['season'] = df['date'].dt.month.apply(
        lambda m: "Dry Season" if m in [12, 1, 2, 3, 4, 5] else "Rainy Season"
    )
    return df

def generate_forecast_dates(months, num_years=1):
    forecast_dates = []
    today = datetime.today()
    current_year = today.year
    for y in range(num_years):
        for m in months:
            forecast_date = datetime(current_year + y, m, 1)
            if forecast_date > today:
                forecast_dates.append(forecast_date)
    return forecast_dates

def seasonal_forecast(df, months, label, scale_factor=1.0):
    df = df.copy()
    df['days_since'] = (df['date'] - df['date'].min()).dt.days

    x = df[['days_since']].values
    y_sales = df[['total_php']].values
    y_quantity = df[['quantity']].values
    model_sales = LinearRegression().fit(x, y_sales)
    model_quantity = LinearRegression().fit(x, y_quantity)

    # Historical predictions
    historical_predictions = []
    for month in months:
        historical_data = df[df['date'].dt.month == month]
        if len(historical_data) > 0:
            forecast_date = historical_data['date'].min()
            days_since = (forecast_date - df['date'].min()).days
            prediction_sales = model_sales.predict([[days_since]])[0][0] * scale_factor
            prediction_quantity = model_quantity.predict([[days_since]])[0][0]
            historical_predictions.append({
                "date": forecast_date.strftime('%Y-%m-%d'),
                "forecast_sales": round(prediction_sales, 2),
                "forecast_quantity": round(prediction_quantity, 2)
            })

    # Daily forecasts for next 6 months of the season
    last_date = df['date'].max()
    start_date = last_date + pd.Timedelta(days=1)
    end_date = start_date + pd.DateOffset(months=6)
    future_dates = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1))
    future_dates = [d for d in future_dates if d.month in months]

    future_predictions = []
    total_sales = 0
    total_quantity = 0

    for forecast_date in future_dates:
        days_since = (forecast_date - df['date'].min()).days
        prediction_sales = model_sales.predict([[days_since]])[0][0] * scale_factor
        prediction_quantity = model_quantity.predict([[days_since]])[0][0]

        prediction_sales = max(0, prediction_sales)
        prediction_quantity = max(0, prediction_quantity)

        future_predictions.append({
            "date": forecast_date.strftime('%Y-%m-%d'),
            "forecast_sales": round(prediction_sales, 2),
            "forecast_quantity": round(prediction_quantity, 2)
        })
        total_sales += prediction_sales
        total_quantity += prediction_quantity

    trend_sales = "Increasing" if total_sales > df[df['date'].dt.month.isin(months)]['total_php'].sum() else "Decreasing"
    trend_quantity = "Increasing" if total_quantity > df[df['date'].dt.month.isin(months)]['quantity'].sum() else "Decreasing"

    return {
        "summary": {
            "label": label,
            "forecast_sales": round(total_sales, 2),
            "forecast_quantity": round(total_quantity, 2),
            "trend_sales": trend_sales,
            "trend_quantity": trend_quantity
        },
        "historical_predictions": historical_predictions,
        "future_predictions": future_predictions
    }

# === API ROUTES ===
@app.route('/forecast', methods=['GET'])
def forecast_api():
    df = get_sales_data()

    dry_months = [12, 1, 2, 3, 4, 5]
    rainy_months = [6, 7, 8, 9, 10, 11]

    dry_result = seasonal_forecast(df, dry_months, "🌞 Dry Season")
    rainy_result = seasonal_forecast(df, rainy_months, "🌧️ Rainy Season")

    results = {
        "forecast_data": [
            dry_result["summary"],
            rainy_result["summary"]
        ],
        "dry_season_data": {
            "dates": df[df['season'] == "Dry Season"]['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": df[df['season'] == "Dry Season"]['total_php'].tolist(),
            "quantity": df[df['season'] == "Dry Season"]['quantity'].tolist()
        },
        "rainy_season_data": {
            "dates": df[df['season'] == "Rainy Season"]['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": df[df['season'] == "Rainy Season"]['total_php'].tolist(),
            "quantity": df[df['season'] == "Rainy Season"]['quantity'].tolist()
        },
        "dry_forecast_monthly": dry_result["future_predictions"],
        "rainy_forecast_monthly": rainy_result["future_predictions"],
        "dry_historical_predictions": dry_result["historical_predictions"],
        "rainy_historical_predictions": rainy_result["historical_predictions"]
    }

    return jsonify(results)

@app.route('/forecast-units', methods=['GET'])
def forecast_units_api():
    # This is to fetch forecast units by category, similar to how it is done in sales data forecasting
    df = get_sales_data()

    results = {}

    for season in ['Dry Season', 'Rainy Season']:
    season_df = df[df['season'] == season]
    season_result = []

    for category in season_df['category'].unique():
        cat_df = season_df[season_df['category'] == category]
        cat_df = cat_df.copy()
        cat_df['days_since'] = (cat_df['date'] - cat_df['date'].min()).dt.days
        x = cat_df[['days_since']].values
        y_quantity = cat_df[['quantity']].values
        y_sales = cat_df[['total_php']].values
        model_quantity = LinearRegression().fit(x, y_quantity)
        model_sales = LinearRegression().fit(x, y_sales)

        # Forecast for next 12 months
        last_date = cat_df['date'].max()
        forecast_months = []
        current = last_date.replace(day=1)

        while len(forecast_months) < 12:
            current = current + pd.DateOffset(months=1)
            if current.month in [12, 1, 2, 3, 4, 5]:  # Adjust for the relevant months
                forecast_months.append(current)

        category_result = []
        total_quantity = 0
        total_sales = 0

        for forecast_date in forecast_months:
            days_since = (forecast_date - cat_df['date'].min()).days
            forecast_quantity = model_quantity.predict([[days_since]])[0][0]
            forecast_sales = model_sales.predict([[days_since]])[0][0]

            forecast_quantity = max(0, forecast_quantity)
            forecast_sales = max(0, forecast_sales)

            category_result.append({
                "category": category,
                "forecast_quantity": round(forecast_quantity, 2),
                "forecast_sales": round(forecast_sales, 2)
            })
            total_quantity += forecast_quantity
            total_sales += forecast_sales

        results[season] = category_result


    return jsonify(results)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
