import os
from dotenv import load_dotenv
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import psycopg2
import urllib.parse as up
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time 

# 1️⃣ Ładowanie .env
load_dotenv()
INFLUX_URL = os.getenv("INFLUXDB_URL")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = os.getenv("INFLUXDB_ORG")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL")

if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, DATABASE_URL]):
    print("❌ Brakuje konfiguracji!")
    exit()

# 2️⃣ InfluxDB Client
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
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Błąd zapytania: {e}")
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

# 3️⃣ Pobierz dane z 7 dni
now = datetime.utcnow()
start_time = (now - timedelta(days=30)).isoformat() + "Z"
end_time = now.isoformat() + "Z"
print(f"Pobieranie danych od {start_time} do {end_time}...")
df = fetch_data(start_time, end_time)

if df.empty:
    print("⚠️ Brak danych!")
    exit()
print(f"✅ Dane pobrane: {len(df)} rekordów")

# 4️⃣ Przygotowanie danych
df = df.dropna()
X = df[["temperature", "humidity", "lux", "output_current"]]
y = df["charging_current"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5️⃣ Trenowanie modelu
print("🚀 Trenuję model do charging_current...")
start = time.time()
model = LinearRegression()

model.fit(X_train, y_train)
end = time.time()
train_duration = end - start
print(f"✅ Model gotowy. Czas trenowania: {train_duration:.2f} sekundy")


# 6️⃣ Ewaluacja
y_pred = model.predict(X_test)
print(f"📉 MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"📈 R2 Score: {r2_score(y_test, y_pred):.4f}")

# 7️⃣ Zapis do Neon
print("💾 Zapisuję model i scaler do Neon...")
model_bytes = pickle.dumps(model)
scaler_bytes = pickle.dumps(scaler)

try:
    conn = get_neon_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO models (name, model_data) VALUES (%s, %s)", ("model_charging_linear", psycopg2.Binary(model_bytes)))
    cursor.execute("INSERT INTO models (name, model_data) VALUES (%s, %s)", ("model_charging_linear_scaler", psycopg2.Binary(scaler_bytes)))
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Zapisano model_charging i scaler.")
except Exception as e:
    print(f"❌ Błąd zapisu: {e}")




plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Wartości rzeczywiste (y_test)")
plt.ylabel("Wartości przewidziane (y_pred)")
plt.title("🎯 Predykcja vs Rzeczywistość")
plt.grid(True)
plt.tight_layout()
plt.show()


residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0, linestyle='--', color='r')
plt.xlabel("Przewidywana wartość")
plt.ylabel("Błąd (residual)")
plt.title("🧪 Błąd predykcji vs przewidywana wartość")
plt.grid(True)
plt.tight_layout()
plt.show()
