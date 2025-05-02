from flask import Flask, jsonify, render_template
from flask_cors import CORS
import os
import json
import pandas as pd
from datetime import datetime
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

def seasonal_monthly_forecast(df, months, label, scale_factor=1.0):
    today = datetime.today()
    year = today.year

    if months[0] == 12:
        start_year = year if today.month < 12 else year + 1
        forecast_dates = [datetime(start_year, 12, 1)]
        for m in [1, 2, 3, 4, 5]:
            forecast_dates.append(datetime(start_year + 1, m, 1))
    else:
        start_year = year if today.month < 6 else year + 1
        forecast_dates = [datetime(start_year, m, 1) for m in months]

    monthly_predictions = []
    total = 0

    for forecast_date in forecast_dates:
        month = forecast_date.month
        label_date = forecast_date.strftime('%Y-%m')

        month_df = df[df['date'].dt.month == month]
        if len(month_df) == 0:
            monthly_predictions.append({
                "date": label_date + "-01",
                "forecast_sales": None
            })
            continue

        month_df = month_df.copy()
        base_date = month_df['date'].min()
        month_df['days_since'] = (month_df['date'] - base_date).dt.days

        if len(month_df) == 1:
            # Use that single value as fallback
            prediction = month_df['total_php'].iloc[0] * scale_factor
        else:
            x = month_df[['days_since']].values
            y = month_df[['total_php']].values
            model = LinearRegression().fit(x, y)
            forecast_day = (forecast_date - base_date).days
            prediction = model.predict([[forecast_day]])[0][0] * scale_factor

        monthly_predictions.append({
            "date": label_date + "-01",
            "forecast_sales": round(prediction, 2)
        })
        total += prediction

    last_values = df[df['date'].dt.month.isin(months)].sort_values('date')['total_php'].values
    last_actual = last_values[-1] if len(last_values) > 0 else 0
    trend = "Increasing" if total > last_actual else "Decreasing" if total < last_actual else "Flat"

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

    dry_result = seasonal_monthly_forecast(df, dry_months, "🌞 Dry Season", scale_factor=1.0)
    rainy_result = seasonal_monthly_forecast(df, rainy_months, "🌧️ Rainy Season", scale_factor=1.0)

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
