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

    # Resample data by month to reduce points (Monthly data)
    df = df.resample('M', on='date').agg({'total_php': 'sum'}).reset_index()

    # Apply Moving Average (e.g., 3-month moving average) for smoothing
    df['moving_avg'] = df['total_php'].rolling(window=3).mean()

    # Add seasonal adjustment based on historical trends
    df['season'] = df['date'].dt.month.apply(
        lambda m: "Dry Season" if m in [12, 1, 2, 3, 4, 5] else "Rainy Season"
    )

    return df

def adjust_forecast(df, label, scale_factor=1.0):
    if df.empty or len(df) < 2:
        return { "label": label, "error": "Not enough data." }

    # Use moving average or linear regression for forecast adjustment
    # Adjust forecast based on moving average and seasonality

    last_actual = df['total_php'].iloc[-1]
    avg_sales = df['moving_avg'].iloc[-1] if not pd.isna(df['moving_avg'].iloc[-1]) else last_actual
    forecast_sales = avg_sales * scale_factor

    trend = "Increasing" if forecast_sales > last_actual else "Decreasing" if forecast_sales < last_actual else "Flat"

    return {
        "label": label,
        "forecast_sales": round(forecast_sales, 2),
        "trend": trend
    }

# === API ROUTES ===
@app.route('/forecast', methods=['GET'])
def forecast_api():
    df = get_sales_data()
    dry_df = df[df['season'] == "Dry Season"]
    rainy_df = df[df['season'] == "Rainy Season"]

    # Prepare forecast data
    dry_forecast = adjust_forecast(dry_df, "🌞 Dry Season", scale_factor=1.1)  # Increased slightly for dry season
    rainy_forecast = adjust_forecast(rainy_df, "🌧️ Rainy Season", scale_factor=0.9)  # Slightly reduced for rainy season
    next_month_forecast = adjust_forecast(df, "📅 Next Month")

    results = {
        "historical_data": {
            "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": df['total_php'].tolist(),
            "moving_avg": df['moving_avg'].tolist()  # Include moving averages
        },
        "forecast_data": [
            dry_forecast,
            rainy_forecast,
            next_month_forecast
        ]
    }

    return jsonify(results)

@app.route('/')
def index():
    return render_template('index.html')

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True)
