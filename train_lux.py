import os
from dotenv import load_dotenv
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import psycopg2
import urllib.parse as up
from datetime import datetime, timedelta
import streamlit as st
load_dotenv()
INFLUX_URL = st.secrets["INFLUXDB_URL"]
INFLUX_TOKEN = st.secrets["INFLUXDB_TOKEN"]
INFLUX_ORG = st.secrets["INFLUXDB_ORG"]
INFLUX_BUCKET = st.secrets["INFLUXDB_BUCKET"]
DATABASE_URL = st.secrets["DATABASE_URL"] 
# 1️⃣ Ładowanie zmiennych środowiskowych
def train_lux():


 

    # 2️⃣ Walidacja konfiguracji
    if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, DATABASE_URL]):
        
        st.error("❌ Brakuje konfiguracji!")
        return

    # 3️⃣ Inicjalizacja klienta InfluxDB
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    # 4️⃣ Pobieranie danych z InfluxDB
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
            return pd.DataFrame()
        except Exception as e:
            
            return pd.DataFrame()

    # 5️⃣ Połączenie z Neon
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

    # 6️⃣ Pobierz dane
    now = datetime.utcnow()
    start_time = (now - timedelta(days=30)).isoformat() + "Z"
    end_time = now.isoformat() + "Z"
    
    df = fetch_data(start_time, end_time)

    if df.empty:
        st.error("❌ Brakuje konfiguracji!")
        return
    

    # 7️⃣ Przygotowanie danych
    df = df.dropna()
    X = df[["temperature", "humidity", "charging_current", "output_current"]]  # cechy
    y = df["lux"]  # przewidywana wartość

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 8️⃣ Trenowanie modelu
   
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
   

    # 9️⃣ Ewaluacja
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    # 🔟 Zapis do Neon
    
    model_bytes = pickle.dumps(model)
    scaler_bytes = pickle.dumps(scaler)

    try:
        conn = get_neon_connection()
        cursor = conn.cursor()
        cursor.execute(
        "UPDATE models SET model_data = %s WHERE name = %s",
        (psycopg2.Binary(model_bytes), "model_lux"))
        cursor.execute(
            "UPDATE models SET model_data = %s WHERE name = %s",
            (psycopg2.Binary(scaler_bytes), "model_lux_scaler"))
        conn.commit()
        cursor.close()
        conn.close()
        st.success("✅ Zapisano model_lux i scaler w Neon.")
    except Exception as e:
        st.error(f"❌ Błąd zapisu do Neon: {e}")
