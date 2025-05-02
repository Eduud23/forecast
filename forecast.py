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
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.resample('M', on='date').agg({'total_php': 'sum'}).reset_index()
    df['days_since'] = (df['date'] - df['date'].min()).dt.days
    df['season'] = df['date'].dt.month.apply(
        lambda m: "Dry Season" if m in [12, 1, 2, 3, 4, 5] else "Rainy Season"
    )
    return df

def forecast(df_subset, label, months_ahead=6, scale_factor=1.0):
    if df_subset.empty or len(df_subset) < 2:
        return { "label": label, "error": "Not enough data." }

    x = df_subset[['days_since']].values
    y = df_subset[['total_php']].values
    model = LinearRegression().fit(x, y)

    forecast_day_start = df_subset['days_since'].max()
    forecast_total = 0

    # Predict for each month in the upcoming season
    for i in range(1, months_ahead + 1):
        forecast_day = forecast_day_start + (30 * i)  # Approximate 30 days/month
        monthly_prediction = model.predict([[forecast_day]])[0][0]
        forecast_total += monthly_prediction

    predicted = forecast_total * scale_factor
    last_actual = y[-1][0]
    trend = "Increasing" if predicted > last_actual else "Decreasing" if predicted < last_actual else "Flat"

    return {
        "label": label,
        "forecast_sales": round(predicted, 2),
        "trend": trend
    }


# === API ROUTES ===
@app.route('/forecast', methods=['GET'])
def forecast_api():
    df = get_sales_data()

    # Filter data for Dry and Rainy seasons
    dry_df = df[df['season'] == "Dry Season"]
    rainy_df = df[df['season'] == "Rainy Season"]

    # Forecast Dry Season sales
    dry_forecast = forecast(dry_df, "🌞 Dry Season", scale_factor=1.5)

    # Forecast Rainy Season sales
    rainy_forecast = forecast(rainy_df, "🌧️ Rainy Season", scale_factor=1.5)

    # Results for each season forecast
    results = {
        "historical_data": {
            "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": df['total_php'].tolist()
        },
        "forecast_data": [
            dry_forecast,
            rainy_forecast
        ],
        "dry_season_data": {
            "dates": dry_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": dry_df['total_php'].tolist()
        },
        "rainy_season_data": {
            "dates": rainy_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": rainy_df['total_php'].tolist()
        }
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

    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    except Exception as e:
        return jsonify({"error": f"Date parsing failed: {str(e)}"}), 500

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
            if len(cat_df) >= 2:
                x = cat_df[['days_since']].values
                y = cat_df[['quantity']].values
                model = LinearRegression().fit(x, y)
                forecast_day = cat_df['days_since'].max() + 30
                predicted_units = model.predict([[forecast_day]])[0][0]
                season_result.append({
                    "category": category,
                    "forecast_units": round(predicted_units, 2)
                })
        results[season] = season_result

    return jsonify(results)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
