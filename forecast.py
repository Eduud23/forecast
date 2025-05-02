from flask import Flask, jsonify, render_template
from flask_cors import CORS
import os
import json
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt
import io
import base64

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
        if 'date' in entry and 'quantity' in entry and 'category' in entry and 'total_php' in entry:
            data.append({
                'date': entry['date'],
                'category': entry['category'],
                'quantity': entry['quantity'],
                'total_php': entry['total_php']
            })

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Aggregate by month
    df['year_month'] = df['date'].dt.to_period('M')
    df = df.groupby(['year_month', 'category']).agg({'total_php': 'sum', 'quantity': 'sum'}).reset_index()
    
    # Add season column (Dry or Rainy)
    df['season'] = df['date'].dt.month.apply(
        lambda m: "Dry Season" if m in [12, 1, 2, 3, 4, 5] else "Rainy Season"
    )

    return df

def forecast(df_subset, label, forecast_months=6):
    if df_subset.empty or len(df_subset) < 2:
        return {"label": label, "error": "Not enough data."}

    # Use months as the x-axis for linear regression
    df_subset['month_since'] = (df_subset['year_month'].dt.month - df_subset['year_month'].min().month) + 1
    x = df_subset[['month_since']].values
    y = df_subset[['total_php']].values
    model = LinearRegression().fit(x, y)

    # Forecast for the next 6 months (approximately)
    forecast_month = df_subset['month_since'].max() + forecast_months
    predicted = model.predict([[forecast_month]])[0][0]

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

    # Forecast for the Dry and Rainy seasons dynamically for the next 6 months
    dry_df = df[df['season'] == "Dry Season"]
    rainy_df = df[df['season'] == "Rainy Season"]

    dry_forecast = forecast(dry_df, "🌞 Dry Season", forecast_months=6)
    rainy_forecast = forecast(rainy_df, "🌧️ Rainy Season", forecast_months=6)

    results = {
        "historical_data": {
            "dates": df['year_month'].dt.strftime('%Y-%m').tolist(),
            "categories": df['category'].tolist(),
            "quantities": df['quantity'].tolist(),
            "totals": df['total_php'].tolist()
        },
        "forecast_data": [
            dry_forecast,
            rainy_forecast
        ],
        "dry_season_specific_data": {
            "dates": dry_df['year_month'].dt.strftime('%Y-%m').tolist(),
            "categories": dry_df['category'].tolist(),
            "quantities": dry_df['quantity'].tolist(),
            "totals": dry_df['total_php'].tolist()
        },
        "rainy_season_specific_data": {
            "dates": rainy_df['year_month'].dt.strftime('%Y-%m').tolist(),
            "categories": rainy_df['category'].tolist(),
            "quantities": rainy_df['quantity'].tolist(),
            "totals": rainy_df['total_php'].tolist()
        }
    }

    return jsonify(results)

@app.route('/forecast_category_trends', methods=['GET'])
def forecast_category_trends():
    season = request.args.get('season', default='Dry Season')
    df = get_sales_data()

    # Filter data by season
    df_season = df[df['season'] == season]

    # Create a trend for each category
    trends = []
    for category in df_season['category'].unique():
        cat_df = df_season[df_season['category'] == category]
        cat_forecast = forecast(cat_df, f"Trend for {category} in {season}")
        trends.append(cat_forecast)

    return jsonify({"trends": trends})

@app.route('/chart/<season>', methods=['GET'])
def show_chart(season):
    df = get_sales_data()
    season_data = df[df['season'] == season]

    # Create a plot for category trends
    plt.figure(figsize=(10, 6))
    for category in season_data['category'].unique():
        category_data = season_data[season_data['category'] == category]
        plt.plot(category_data['year_month'].astype(str), category_data['total_php'], label=category)

    plt.title(f"Sales Trends for {season}")
    plt.xlabel("Month")
    plt.ylabel("Total Sales (PHP)")
    plt.xticks(rotation=45)
    plt.legend()

    # Save the plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return render_template("chart.html", chart=image_base64)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
