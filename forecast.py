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

    # Predict sales for every day in the time span
    forecast_days = pd.DataFrame({
        'days_since': range(df_subset['days_since'].min(), df_subset['days_since'].max() + 31)  # Predict for the next 30 days
    })
    forecast_sales = model.predict(forecast_days[['days_since']])

    forecast_sales = forecast_sales * scale_factor  # Apply scale factor for dry/rainy season
    
    return {
        "label": label,
        "days_since": forecast_days['days_since'].tolist(),
        "forecast_sales": forecast_sales.tolist()
    }

# === API ROUTE ===
@app.route('/forecast', methods=['GET'])
def forecast_api():
    df = get_sales_data()
    dry_df = df[df['season'] == "Dry Season"]
    rainy_df = df[df['season'] == "Rainy Season"]

    # Get historical data for plotting
    historical_data = {
        "dates": df['date'].dt.strftime('%Y-%m-%d').tolist(),
        "sales": df['total_php'].tolist()
    }

    # Forecast data
    results = [
        forecast(dry_df, "🌞 Dry Season", scale_factor=1.5),
        forecast(rainy_df, "🌧️ Rainy Season", scale_factor=1.5),
        forecast(df, "📅 Next Month")
    ]
    return jsonify({
        "historical_data": historical_data,
        "forecast_data": results
    })

# === RENDER THE HTML PAGE WITH CHART ===
@app.route('/')
def index():
    return render_template('index.html')

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True)
