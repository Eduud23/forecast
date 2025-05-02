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

        # Check if 'date' field exists and is a valid string in the expected format
        if 'date' in entry and isinstance(entry['date'], str):
            try:
                # Convert the date string to a datetime object (format: "YYYY-MM-DD")
                entry['date'] = pd.to_datetime(entry['date'], format='%Y-%m-%d', errors='coerce')

                # Only append valid dates
                if pd.notna(entry['date']):
                    data.append({
                        'date': entry['date'],
                        'category': entry['category'],
                        'quantity': entry['quantity'],
                        'total_php': entry['total_php']
                    })
                else:
                    print(f"Invalid date format for document {doc.id}: {entry['date']}")
            except Exception as e:
                print(f"Error converting date for document {doc.id}: {e}")
        else:
            print(f"Missing or invalid date in document {doc.id}")

    # Check if any valid data is found
    if not data:
        raise ValueError("No valid sales data found in Firestore.")

    # Create DataFrame from the gathered data
    df = pd.DataFrame(data)

    # Ensure that the 'date' column is of datetime type and drop rows with invalid dates
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates

    # Check if 'date' column exists
    if 'date' not in df.columns:
        raise ValueError("The 'date' column is missing from the DataFrame")

    # Aggregate by month
    df['month'] = df['date'].dt.to_period('M')

    # Ensure 'month' column is created
    if 'month' not in df.columns:
        raise ValueError("The 'month' column could not be created")

    df = df.groupby(['month', 'category']).agg({'total_php': 'sum', 'quantity': 'sum'}).reset_index()

    # Add season column (Dry or Rainy)
    df['season'] = df['date'].dt.month.apply(
        lambda m: "Dry Season" if m in [12, 1, 2, 3, 4, 5] else "Rainy Season"
    )

    return df

def forecast(df_subset, label, forecast_months=6):
    if df_subset.empty or len(df_subset) < 2:
        return {"label": label, "error": "Not enough data."}

    # Convert month periods to numeric values for regression
    df_subset['month_numeric'] = df_subset['month'].dt.month
    x = df_subset[['month_numeric']].values
    y = df_subset[['total_php']].values

    # Train the linear regression model
    model = LinearRegression().fit(x, y)

    # Forecast for the next 6 months (approximately 180 days)
    forecast_month = df_subset['month_numeric'].max() + forecast_months
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
    try:
        df = get_sales_data()

        # Forecast for the Dry and Rainy seasons dynamically for the next 6 months
        dry_df = df[df['season'] == "Dry Season"]
        rainy_df = df[df['season'] == "Rainy Season"]

        dry_forecast = forecast(dry_df, "🌞 Dry Season", forecast_months=6)
        rainy_forecast = forecast(rainy_df, "🌧️ Rainy Season", forecast_months=6)

        results = {
            "historical_data": {
                "dates": df['month'].dt.strftime('%Y-%m').tolist(),
                "categories": df['category'].tolist(),
                "quantities": df['quantity'].tolist(),
                "totals": df['total_php'].tolist()
            },
            "forecast_data": [
                dry_forecast,
                rainy_forecast
            ],
            "dry_season_specific_data": {
                "dates": dry_df['month'].dt.strftime('%Y-%m').tolist(),
                "categories": dry_df['category'].tolist(),
                "quantities": dry_df['quantity'].tolist(),
                "totals": dry_df['total_php'].tolist()
            },
            "rainy_season_specific_data": {
                "dates": rainy_df['month'].dt.strftime('%Y-%m').tolist(),
                "categories": rainy_df['category'].tolist(),
                "quantities": rainy_df['quantity'].tolist(),
                "totals": rainy_df['total_php'].tolist()
            }
        }

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/category_trends/<season>', methods=['GET'])
def category_trends(season):
    try:
        df = get_sales_data()
        df_season = df[df['season'] == season]

        trends = []
        for category in df_season['category'].unique():
            cat_df = df_season[df_season['category'] == category]
            category_trend = {
                'category': category,
                'total_php': cat_df.groupby('month')['total_php'].sum().tolist(),
                'dates': cat_df['month'].dt.strftime('%Y-%m').unique().tolist()
            }
            trends.append(category_trend)

        return jsonify(trends)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
