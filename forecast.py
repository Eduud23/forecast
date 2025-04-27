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

    forecast_day = df_subset['days_since'].max() + 30
    predicted = model.predict([[forecast_day]])[0][0] * scale_factor

    last_actual = y[-1][0]
    trend = "Increasing" if predicted > last_actual else "Decreasing" if predicted < last_actual else "Flat"

    return {
        "label": label,
        "forecast_sales": round(predicted, 2),
        "trend": trend
    }

# === API ROUTE ===
@app.route('/forecast', methods=['GET'])
def forecast_api():
    df = get_sales_data()
    dry_df = df[df['season'] == "Dry Season"]
    rainy_df = df[df['season'] == "Rainy Season"]

    results = [
        forecast(dry_df, "🌞 Dry Season", scale_factor=1.5),
        forecast(rainy_df, "🌧️ Rainy Season", scale_factor=1.5),
        forecast(df, "📅 Next Month")
    ]
    return jsonify(results)

# === NEW API ROUTE ===
@app.route('/get_monthly_sales_for_graph', methods=['GET'])
def get_monthly_sales_for_graph():
    try:
        # Fetch the raw sales data
        df = get_sales_data()

        # Prepare monthly aggregated data
        df['month_year'] = df['date'].dt.to_period('M')

        # Convert Period to string for JSON serialization
        df['month_year'] = df['month_year'].astype(str)

        df_monthly = df.groupby('month_year').agg({'total_php': 'sum'}).reset_index()

        # Prepare forecasted data for dry season, rainy season, and overall
        dry_df = df[df['month_year'].dt.month.isin([12, 1, 2, 3, 4, 5])]
        rainy_df = df[df['month_year'].dt.month.isin([6, 7, 8, 9, 10, 11])]

        dry_forecast = forecast(dry_df, "🌞 Dry Season", scale_factor=1.5)
        rainy_forecast = forecast(rainy_df, "🌧️ Rainy Season", scale_factor=1.5)
        all_forecast = forecast(df, "📅 Next Month")

        # Create a new DataFrame for the next month's forecasted data
        new_row = pd.DataFrame([{
            'month_year': str(df_monthly['month_year'].max() + 1),  # Next month as string
            'total_php': all_forecast['forecast_sales'],
            'forecasted_sales': all_forecast['forecast_sales']
        }])

        # Concatenate the new row to the existing df_monthly
        df_monthly = pd.concat([df_monthly, new_row], ignore_index=True)

        # Return the monthly sales data along with forecasted data
        return jsonify({
            "monthly_sales": df_monthly.to_dict(orient="records"),
            "dry_season_forecast": dry_forecast,
            "rainy_season_forecast": rainy_forecast,
            "all_data_forecast": all_forecast
        })
    except Exception as e:
        print(f"Error in get_monthly_sales_for_graph: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


# === RENDER THE HTML PAGE WITH CHART ===
@app.route('/')
def index():
    return render_template('index.html')

# === MAIN ===
if __name__ == '__main__':
    app.run(debug=True)
