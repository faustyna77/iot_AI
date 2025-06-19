import os
from dotenv import load_dotenv
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import psycopg2
import urllib.parse as up
from datetime import datetime, timedelta
import pickle
import streamlit as st

load_dotenv()

INFLUX_URL = st.secrets["INFLUXDB_URL"]
INFLUX_TOKEN = st.secrets["INFLUXDB_TOKEN"]
INFLUX_ORG = st.secrets["INFLUXDB_ORG"]
INFLUX_BUCKET = st.secrets["INFLUXDB_BUCKET"]
DATABASE_URL = st.secrets["DATABASE_URL"] 

# 🧹 Czyszczenie danych
def remove_anomalies(df, features, contamination=0.05):
    df_clean = df.copy().dropna()
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(df_clean[features])
    df_clean['anomaly'] = preds
    df_filtered = df_clean[df_clean['anomaly'] == 1].drop(columns=['anomaly'])
    return df_filtered

def get_neon_connection():
    up.uses_netloc.append("postgres")
    url = up.urlparse(DATABASE_URL)
    return psycopg2.connect(
        dbname=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port
    )

def fetch_data(start_time, end_time):
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: {start_time}, stop: {end_time})
            |> filter(fn: (r) => r["_measurement"] == "env_measurements")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        result = client.query_api().query_data_frame(query)
        if not result.empty:
            result['_time'] = pd.to_datetime(result['_time'])
            return result[['_time', 'temperature', 'humidity', 'lux', 'output_current', 'charging_current']]
    except Exception as e:
        st.error(f"❌ Błąd pobierania danych: {e}")
    return pd.DataFrame()

def train_isolation():
    if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, DATABASE_URL]):
        st.error("❌ Brakuje konfiguracji środowiska!")
        return

    # ⏱️ Pobierz dane z 30 dni
    now = datetime.utcnow()
    start_time = (now - timedelta(days=30)).isoformat() + "Z"
    end_time = now.isoformat() + "Z"
    df = fetch_data(start_time, end_time)

    if df.empty:
        st.error("❌ Brak danych z InfluxDB.")
        return

    st.success(f"✅ Dane pobrane: {len(df)} rekordów")

    # 🔍 Czyszczenie danych
    features = ['temperature', 'humidity', 'lux', 'charging_current']
    df_cleaned = remove_anomalies(df, features)

    # 🧼 Usuń wartości ujemne i ekstremalne
    df_cleaned = df_cleaned[df_cleaned['output_current'] >= 0]
    mean = df_cleaned['output_current'].mean()
    std = df_cleaned['output_current'].std()
    df_cleaned = df_cleaned[
        (df_cleaned['output_current'] >= mean - 3 * std) &
        (df_cleaned['output_current'] <= mean + 3 * std)
    ]

    st.info(f"📊 Dane po oczyszczeniu: {len(df_cleaned)} rekordów")

    if df_cleaned.empty:
        st.error("❌ Wszystkie dane zostały odrzucone jako nieprawidłowe.")
        return

    # 🧠 Trenowanie modelu
    X = df_cleaned[features]
    y = df_cleaned['output_current']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.success(f"📉 MSE: {mean_squared_error(y_test, y_pred):.4f}")
    st.success(f"📈 R2 Score: {r2_score(y_test, y_pred):.4f}")

    # 💾 Zapisz model i scaler
    model_bytes = pickle.dumps(model)
    scaler_bytes = pickle.dumps(scaler)

    try:
        conn = get_neon_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE models SET model_data = %s WHERE name = %s",
            (psycopg2.Binary(model_bytes), "model_v1_anomalion_off"))
        cursor.execute(
            "UPDATE models SET model_data = %s WHERE name = %s",
            (psycopg2.Binary(scaler_bytes), "model_v1_anomalion_off_scaler"))

        conn.commit()
        cursor.close()
        conn.close()
        st.success("✅ Model i scaler zostały zapisane do Neon.")
    except Exception as e:
        st.error(f"❌ Błąd zapisu do Neon: {e}")
