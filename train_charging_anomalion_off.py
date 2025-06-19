import os
from dotenv import load_dotenv
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import psycopg2
import urllib.parse as up
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

import time
import streamlit as st

load_dotenv()
INFLUX_URL = st.secrets["INFLUXDB_URL"]
INFLUX_TOKEN = st.secrets["INFLUXDB_TOKEN"]
INFLUX_ORG = st.secrets["INFLUXDB_ORG"]
INFLUX_BUCKET = st.secrets["INFLUXDB_BUCKET"]
DATABASE_URL = st.secrets["DATABASE_URL"] 

def train_charging_anomalion_off():
    if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, DATABASE_URL]):
        st.error("❌ Brakuje konfiguracji!")
        return

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    def fetch_data(start_time, end_time):
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: {start_time}, stop: {end_time})
            |> filter(fn: (r) => r["_measurement"] == "env_measurements")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        try:
            df = client.query_api().query_data_frame(query)
            if not df.empty:
                df["_time"] = pd.to_datetime(df["_time"])
                return df[["_time", "temperature", "humidity", "lux", "output_current", "charging_current"]]
        except Exception as e:
            st.error(f"❌ Błąd pobierania danych: {e}")
        return pd.DataFrame()

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

    # 📥 Pobierz dane z 30 dni
    now = datetime.utcnow()
    start_time = (now - timedelta(days=30)).isoformat() + "Z"
    end_time = now.isoformat() + "Z"
    df = fetch_data(start_time, end_time)

    if df.empty:
        st.error("❌ Brak danych do treningu.")
        return

    st.success(f"✅ Dane pobrane: {len(df)} rekordów")

    # 🧹 Czyszczenie danych
    df = df.dropna()
    df = df[df["charging_current"] >= 0]  # usuń ujemne wartości

    # usuń wartości odstające na podstawie 3σ
    mean_c = df["charging_current"].mean()
    std_c = df["charging_current"].std()
    df = df[(df["charging_current"] >= mean_c - 3 * std_c) & (df["charging_current"] <= mean_c + 3 * std_c)]

    # IsolationForest na cechach wejściowych
    features = ["temperature", "humidity", "lux", "output_current"]
    try:
        iso = IsolationForest(contamination=0.05, random_state=42)
        mask = iso.fit_predict(df[features]) == 1
        df = df[mask]
        st.success(f"📉 Dane po czyszczeniu i usunięciu anomalii: {len(df)} rekordów")
    except Exception as e:
        st.success(f"⚠️ Błąd detekcji anomalii: {e}")

    if df.empty:
        st.error("❌ Wszystkie dane zostały usunięte jako nieprawidłowe.")
        return

    # 🧠 Trenowanie modelu
    X = df[features]
    y = df["charging_current"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    start = time.time()
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    end = time.time()

    train_duration = end - start
    st.success(f"✅ Model gotowy. Czas trenowania: {train_duration:.2f} sekundy")

    # 📊 Ewaluacja
    y_pred = model.predict(X_test)
    st.success(f"📉 MSE: {mean_squared_error(y_test, y_pred):.4f}")
    st.success(f"📈 R2 Score: {r2_score(y_test, y_pred):.4f}")

    # 💾 Zapis do Neon
    model_bytes = pickle.dumps(model)
    scaler_bytes = pickle.dumps(scaler)

    try:
        conn = get_neon_connection()
        cursor = conn.cursor()
        cursor.execute(
        "UPDATE models SET model_data = %s WHERE name = %s",
        (psycopg2.Binary(model_bytes), "model_charging_anomalion_off"))
        cursor.execute(
        "UPDATE models SET model_data = %s WHERE name = %s",
        (psycopg2.Binary(scaler_bytes), "model_charging_anomalion_off_scaler"))
        conn.commit()
        cursor.close()
        conn.close()
        st.success("✅ Zapisano model_charging i scaler.")
    except Exception as e:
        st.error(f"❌ Błąd zapisu: {e}")

