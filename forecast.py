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

def forecast(df_subset, label, scale_factor=1.0):
    if df_subset.empty or len(df_subset) < 2:
        return { "label": label, "error": "Not enough data." }

    x = df_subset[['days_since']].values
    y = df_subset[['total_php']].values
    model = LinearRegression().fit(x, y)

    forecast_day = df_subset['days_since'].max() + 30
    predicted = model.predict([[forecast_day]])[0][0] * scale_factor

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
    dry_df = df[df['season'] == "Dry Season"]
    rainy_df = df[df['season'] == "Rainy Season"]

    # Dry and Rainy Season Forecasts (Dry season: December to May, Rainy season: June to November)
    dry_df_specific = dry_df[(dry_df['date'] >= datetime(2024, 12, 1)) & (dry_df['date'] <= datetime(2025, 5, 31))]
    rainy_df_specific = rainy_df[(rainy_df['date'] >= datetime(2025, 6, 1)) & (rainy_df['date'] <= datetime(2025, 11, 30))]

    dry_forecast = forecast(dry_df_specific, "🌞 Dry Season", scale_factor=1.5)
    rainy_forecast = forecast(rainy_df_specific, "🌧️ Rainy Season", scale_factor=1.5)

    # Next Month Forecast (Sales)
    today = datetime.today()
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
        ],
        "dry_season_specific_data": {
            "dates": dry_df_specific['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": dry_df_specific['total_php'].tolist()
        },
        "rainy_season_specific_data": {
            "dates": rainy_df_specific['date'].dt.strftime('%Y-%m-%d').tolist(),
            "sales": rainy_df_specific['total_php'].tolist()
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

    # === Next Month Forecast (Units) ===
    today = datetime.today()
    next_month_start = datetime(today.year + (today.month // 12), (today.month % 12) + 1, 1)
    next_month_end = (next_month_start + pd.offsets.MonthEnd(1)).to_pydatetime()

    forecast_day = (next_month_start - df['date'].min()).days
    next_month_results = []

    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        if len(cat_df) >= 2:
            x = cat_df[['days_since']].values
            y = cat_df[['quantity']].values
            model = LinearRegression().fit(x, y)
            predicted_units = model.predict([[forecast_day]])[0][0]
            next_month_results.append({
                "category": category,
                "forecast_units": round(predicted_units, 2)
            })

    results["Next Month Forecast Range"] = {
        "start_date": next_month_start.strftime("%Y-%m-%d"),
        "end_date": next_month_end.strftime("%Y-%m-%d")
    }
    results["Next Month"] = next_month_results

    return jsonify(results)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
