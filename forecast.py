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
        if 'date' in entry and 'total_php' in entry:
            data.append({
                'date': entry['date'],
                'total_php': entry['total_php']
            })

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')
    df = df.resample('M', on='date').agg({'total_php': 'sum'}).reset_index()
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

def seasonal_monthly_forecast(df, months, label, scale_factor=1.0):
    # Prepare full model
    df = df.copy()
    df['days_since'] = (df['date'] - df['date'].min()).dt.days

    x = df[['days_since']].values
    y = df[['total_php']].values
    model = LinearRegression().fit(x, y)

    # Forecast next 2 seasonal cycles (e.g., 12 dry or rainy months)
    last_date = df['date'].max()
    forecast_months = []
    current = last_date.replace(day=1)

    while len(forecast_months) < 12:
        current = current + pd.DateOffset(months=1)
        if current.month in months:
            forecast_months.append(current)

    monthly_predictions = []
    total = 0

    for forecast_date in forecast_months:
        days_since = (forecast_date - df['date'].min()).days
        prediction = model.predict([[days_since]])[0][0] * scale_factor
        prediction = max(0, prediction)  # ensure no negative forecasts

        monthly_predictions.append({
            "date": forecast_date.strftime('%Y-%m-%d'),
            "forecast_sales": round(prediction, 2)
        })
        total += prediction

    trend = "Increasing" if total > df[df['date'].dt.month.isin(months)]['total_php'].sum() else "Decreasing"

    return {
        "summary": {
            "label": label,
            "forecast_sales": round(total, 2),
            "trend": trend
        },
        "monthly": monthly_predictions
    }

# === API ROUTES ===
@app.route('/forecast', methods=['GET'])
def forecast_api():
    df = get_sales_data()

    dry_months = [12, 1, 2, 3, 4, 5]
    rainy_months = [6, 7, 8, 9, 10, 11]

    dry_result = seasonal_monthly_forecast(df, dry_months, "🌞 Dry Season")
    rainy_result = seasonal_monthly_forecast(df, rainy_months, "🌧️ Rainy Season")

    results = {
        "forecast_data": [
            dry_result["summary"],
            rainy_result["summary"]
        ],
        "dry_season_data": {
            "dates": df[df['season'] == "Dry Season"]['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": df[df['season'] == "Dry Season"]['total_php'].tolist()
        },
        "rainy_season_data": {
            "dates": df[df['season'] == "Rainy Season"]['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": df[df['season'] == "Rainy Season"]['total_php'].tolist()
        },
        "dry_forecast_monthly": dry_result["monthly"],
        "rainy_forecast_monthly": rainy_result["monthly"]
    }

    return jsonify(results)

@app.route('/forecast-units', methods=['GET'])
def forecast_units_api():
    sales_ref = db.collection("sales_orders").order_by("date")
    docs = sales_ref.stream()
    data = []

    for doc in docs:
        entry = doc.to_dict()
        if 'date' in entry and 'quantity' in entry and 'category' in entry:
            data.append({
                'date': entry['date'],
                'quantity': entry['quantity'],
                'category': entry['category']
            })

    if not data:
        return jsonify({"error": "No sales unit data found."}), 400

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if df.empty:
        return jsonify({"error": "No valid sales data after date cleaning."}), 400

    df = df.sort_values('date')
    df['days_since'] = (df['date'] - df['date'].min()).dt.days
    df['season'] = df['date'].dt.month.apply(
        lambda m: "Dry Season" if m in [12, 1, 2, 3, 4, 5] else "Rainy Season"
    )

    results = {}

    for season in ['Dry Season', 'Rainy Season']:
        season_df = df[df['season'] == season]
        season_result = []

        for category in season_df['category'].unique():
            cat_df = season_df[season_df['category'] == category]
            if len(cat_df) == 0:
                continue

            cat_df = cat_df.copy()
            if len(cat_df) == 1:
                forecast_units = cat_df['quantity'].iloc[0]
            else:
                x = cat_df[['days_since']].values
                y = cat_df[['quantity']].values
                model = LinearRegression().fit(x, y)
                forecast_day = cat_df['days_since'].max() + 30
                forecast_units = model.predict([[forecast_day]])[0][0]

            season_result.append({
                "category": category,
                "forecast_units": round(forecast_units, 2)
            })

        results[season] = season_result

    return jsonify(results)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
