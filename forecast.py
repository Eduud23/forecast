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
    raise ValueError("FIREBASE_KEY_JSON environment variable is not set.")

cred = credentials.Certificate(json.loads(firebase_key_json))
firebase_admin.initialize_app(cred)
db = firestore.client()

# === HELPER FUNCTION ===
def get_sales_data():
    sales_ref = db.collection("sales_orders").order_by("date")
    docs = sales_ref.stream()
    data = []

    for doc in docs:
        entry = doc.to_dict()
        if all(k in entry for k in ['date', 'total_php', 'quantity', 'category']):
            data.append({
                'date': entry['date'],
                'total_php': float(entry['total_php']),
                'quantity': int(entry['quantity']),
                'category': entry['category']
            })

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['season'] = df['date'].dt.month.apply(lambda m: 'Dry Season' if m in [12, 1, 2, 3, 4, 5] else 'Rainy Season')
    return df


def forecast_category_trends(df, season_months):
    trend_data = []

    for category in df['category'].unique():
        cat_df = df[(df['category'] == category) & (df['date'].dt.month.isin(season_months))]
        if len(cat_df) < 5:
            continue  # Skip if not enough data

        cat_df = cat_df.copy()
        cat_df['days_since'] = (cat_df['date'] - cat_df['date'].min()).dt.days
        x = cat_df[['days_since']]
        y = cat_df[['quantity']]
        model = LinearRegression().fit(x, y)

        # Forecast next 6 months for this season
        last_date = cat_df['date'].max()
        forecast_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180)
        forecast_days = [d for d in forecast_days if d.month in season_months]
        total_forecast = 0

        for forecast_date in forecast_days:
            days_since = (forecast_date - cat_df['date'].min()).days
            predicted = model.predict(pd.DataFrame({'days_since': [days_since]}))[0]
            total_forecast += max(0, predicted)


        past_total = cat_df['quantity'].sum()
        trend_status = 'Increasing' if total_forecast > past_total else 'Decreasing'

        trend_data.append({
            'category': category,
            'season': 'Dry Season' if season_months[0] in [12, 1, 2, 3, 4, 5] else 'Rainy Season',
            'historical_quantity': float(past_total),
            'forecast_quantity': float(total_forecast),
            'trend': trend_status,
            'dates': [d.strftime('%Y-%m-%d') for d in cat_df['date']],
            'quantities': [int(q) for q in cat_df['quantity']]
        })

    return trend_data

# === API ROUTE ===
@app.route('/category-trends', methods=['GET'])
def category_trends():
    df = get_sales_data()
    dry_months = [12, 1, 2, 3, 4, 5]
    rainy_months = [6, 7, 8, 9, 10, 11]

    dry_trends = forecast_category_trends(df, dry_months)
    rainy_trends = forecast_category_trends(df, rainy_months)

    return jsonify({
        "dry_season_trends": dry_trends,
        "rainy_season_trends": rainy_trends
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
