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
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.resample('M', on='date').agg({'total_php': 'sum'}).reset_index()
    df['days_since'] = (df['date'] - df['date'].min()).dt.days
    df['season'] = df['date'].dt.month.apply(
        lambda m: "Dry Season" if m in [12, 1, 2, 3, 4, 5] else "Rainy Season"
    )
    return df

# === API ROUTES ===
@app.route('/forecast', methods=['GET'])
def forecast_api():
    df = get_sales_data()
    if df.empty:
        return jsonify({"error": "No sales data available"}), 400

    dry_df = df[df['season'] == "Dry Season"]
    rainy_df = df[df['season'] == "Rainy Season"]

    today = datetime.today()
    year = today.year

    # Determine upcoming dry season
    if today.month >= 6:
        dry_start = datetime(year + 1, 1, 1)
        dry_end = datetime(year + 1, 5, 31)
    else:
        dry_start = datetime(year, 12, 1)
        dry_end = datetime(year + 1, 5, 31)

    # Determine upcoming rainy season
    if today.month >= 12 or today.month <= 5:
        rainy_start = datetime(year, 6, 1)
        rainy_end = datetime(year, 11, 30)
    else:
        rainy_start = datetime(year + 1, 6, 1)
        rainy_end = datetime(year + 1, 11, 30)

    dry_model = LinearRegression().fit(dry_df[['days_since']], dry_df[['total_php']])
    rainy_model = LinearRegression().fit(rainy_df[['days_since']], rainy_df[['total_php']])

    dry_forecast_day = (dry_start + timedelta(days=90) - df['date'].min()).days
    rainy_forecast_day = (rainy_start + timedelta(days=90) - df['date'].min()).days

    dry_predicted = dry_model.predict([[dry_forecast_day]])[0][0] * 6  # 6-month total
    rainy_predicted = rainy_model.predict([[rainy_forecast_day]])[0][0] * 6

    dry_forecast = {
        "label": "🌞 Next Dry Season",
        "forecast_sales": round(dry_predicted, 2),
        "trend": "N/A",
        "forecast_range": {
            "start_date": dry_start.strftime("%Y-%m-%d"),
            "end_date": dry_end.strftime("%Y-%m-%d")
        }
    }

    rainy_forecast = {
        "label": "🌧️ Next Rainy Season",
        "forecast_sales": round(rainy_predicted, 2),
        "trend": "N/A",
        "forecast_range": {
            "start_date": rainy_start.strftime("%Y-%m-%d"),
            "end_date": rainy_end.strftime("%Y-%m-%d")
        }
    }

    # === Next Month Forecast (Sales) ===
    next_month_start = datetime(today.year + (today.month // 12), (today.month % 12) + 1, 1)
    next_month_end = (next_month_start + pd.offsets.MonthEnd(1)).to_pydatetime()

    forecast_day = (next_month_start - df['date'].min()).days
    model = LinearRegression().fit(df[['days_since']], df[['total_php']])
    predicted = model.predict([[forecast_day]])[0][0]
    last_actual = df['total_php'].iloc[-1]
    trend = "Increasing" if predicted > last_actual else "Decreasing" if predicted < last_actual else "Flat"

    next_month_forecast = {
        "label": "📅 Next Month",
        "forecast_sales": round(predicted, 2),
        "trend": trend,
        "forecast_range": {
            "start_date": next_month_start.strftime("%Y-%m-%d"),
            "end_date": next_month_end.strftime("%Y-%m-%d")
        }
    }

    results = {
        "historical_data": {
            "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": df['total_php'].tolist()
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

if __name__ == '__main__':
    app.run(debug=True)
