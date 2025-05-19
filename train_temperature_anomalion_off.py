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
INFLUX_URL = os.getenv("INFLUXDB_URL")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = os.getenv("INFLUXDB_ORG")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL") # URL do Neon (Postgres)
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

def train_temperature_anomalion_off():
    if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, DATABASE_URL]):
        st.error("❌ Brakuje konfiguracji!")
        return

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    def fetch_data(start_time, end_time):
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: {start_time}, stop: {end_time})
            |> filter(fn: (r) => r["_measurement"] == "dht_measurements")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        try:
            df = client.query_api().query_data_frame(query)
            if not df.empty:
                df["_time"] = pd.to_datetime(df["_time"])
                return df[["_time", "temperature", "humidity", "lux", "output_current", "charging_current"]]
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Błąd zapytania danych: {e}")
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

    now = datetime.utcnow()
    start_time = (now - timedelta(days=30)).isoformat() + "Z"
    end_time = now.isoformat() + "Z"
    
    df = fetch_data(start_time, end_time)

    if df.empty:
        st.error("❌ Brak danych do treningu.")
        return

    st.success(f"✅ Dane pobrane: {len(df)} rekordów")

    df = df.dropna()
    df = df[df["temperature"] < 35]

    st.info(f"📊 Dane po odfiltrowaniu temperatury > 35°C: {len(df)} rekordów")

    if df.empty:
        st.error("❌ Wszystkie dane zostały odrzucone jako nieprawidłowe.")
        return

    X = df[["humidity", "lux", "output_current", "charging_current"]]
    y = df["temperature"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.success(f"📉 MSE: {mean_squared_error(y_test, y_pred):.4f}")
    st.success(f"📈 R2 Score: {r2_score(y_test, y_pred):.4f}")

    model_bytes = pickle.dumps(model)
    scaler_bytes = pickle.dumps(scaler)

    try:
        conn = get_neon_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE models SET model_data = %s WHERE name = %s",
            ("model_temperature_anomalion_off", psycopg2.Binary(model_bytes))
        )
        cursor.execute(
            "UPDATE models SET model_data = %s WHERE name = %s",
            ("model_temperature_anomalion_off_scaler", psycopg2.Binary(scaler_bytes))
        )
        conn.commit()
        cursor.close()
        conn.close()
        st.success("✅ Model i scaler zostały zapisane do Neon.")
    except Exception as e:
        st.error(f"❌ Błąd zapisu do Neon: {e}")
