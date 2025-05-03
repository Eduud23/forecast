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
        if 'date' in entry and 'total_php' in entry and 'quantity' in entry and 'category' in entry:
            data.append({
                'date': entry['date'],
                'total_php': entry['total_php'],
                'quantity': entry['quantity'],
                'category': entry['category']
            })

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df['season'] = df['date'].dt.month.apply(lambda m: 'Dry Season' if m in [12, 1, 2, 3, 4, 5] else 'Rainy Season')
    return df


def forecast_category_trends(df, season_months):
    trend_data = []

    for category in df['category'].unique():
        cat_df = df[(df['category'] == category) & (df['date'].dt.month.isin(season_months))]
        if len(cat_df) < 5:
            continue  # not enough data to forecast

        cat_df = cat_df.copy()
        cat_df['days_since'] = (cat_df['date'] - cat_df['date'].min()).dt.days
        x = cat_df[['days_since']]
        y = cat_df[['quantity']]
        model = LinearRegression().fit(x, y)

        # Forecast next 6 months
        last_date = cat_df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180)
        future_dates = [d for d in future_dates if d.month in season_months]
        total_forecast = 0

        for date in future_dates:
            days_since = (date - cat_df['date'].min()).days
            predicted_qty = model.predict([[days_since]])[0][0]
            total_forecast += max(predicted_qty, 0)

        past_total = cat_df['quantity'].sum()
        trend_status = 'Increasing' if total_forecast > past_total else 'Decreasing'

        trend_data.append({
            'category': category,
            'season': 'Dry Season' if season_months[0] in [12, 1, 2, 3, 4, 5] else 'Rainy Season',
            'historical_quantity': round(past_total, 2),
            'forecast_quantity': round(total_forecast, 2),
            'trend': trend_status,
            'dates': cat_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            'quantities': cat_df['quantity'].tolist()
        })

    return trend_data


# === API ROUTE TO RETURN CATEGORY TRENDS BY SEASON ===
@app.route('/category-trends', methods=['GET'])
def category_trends():
    df = get_sales_data()

    dry_months = [12, 1, 2, 3, 4, 5]
    rainy_months = [6, 7, 8, 9, 10, 11]

    dry_trends = forecast_category_trends(df, dry_months)
    rainy_trends = forecast_category_trends(df, rainy_months)

    response = {
        "dry_season_trends": dry_trends,
        "rainy_season_trends": rainy_trends
    }

    return jsonify(response)


# === FRONT PAGE ===
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
