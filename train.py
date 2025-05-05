import os
from dotenv import load_dotenv
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import psycopg2
import urllib.parse as up
from datetime import datetime, timedelta
import pickle


# ===== 1️⃣ Ładowanie zmiennych środowiskowych =====

load_dotenv()

INFLUX_URL = os.getenv("INFLUXDB_URL")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = os.getenv("INFLUXDB_ORG")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET")
DATABASE_URL = os.getenv("DATABASE_URL")  # URL do Neon (Postgres)

print("\n=== Konfiguracja InfluxDB ===")
print(f"URL: {INFLUX_URL}")
print(f"ORG: {INFLUX_ORG}")
print(f"BUCKET: {INFLUX_BUCKET}\n")

# ===== 2️⃣ Sprawdzenie poprawności =====

if not all([INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, DATABASE_URL]):
    print("❌ Nie wszystkie zmienne środowiskowe zostały załadowane!")
    exit()

# ===== 3️⃣ Inicjalizacja klienta InfluxDB =====

try:
    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )
except Exception as e:
    print(f"❌ Błąd podczas inicjalizacji klienta InfluxDB: {e}")
    exit()

# ===== 4️⃣ Funkcja do pobierania danych =====

def fetch_data(start_time, end_time):
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {start_time}, stop: {end_time})
        |> filter(fn: (r) => r["_measurement"] == "dht_measurements")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        result = client.query_api().query_data_frame(query)
        if not result.empty:
            result['_time'] = pd.to_datetime(result['_time'])
            return result[['_time', 'temperature', 'humidity', 'lux', 'output_current', 'charging_current']]
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Błąd zapytania danych: {e}")
        return pd.DataFrame()

# ===== 5️⃣ Funkcja do połączenia z Neon =====

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

# ===== 6️⃣ Pobierz dane =====

now = datetime.utcnow()
start_time = (now - timedelta(days=7)).isoformat() + "Z"
end_time = now.isoformat() + "Z"

print(f"Pobieranie danych od {start_time} do {end_time}...")

df = fetch_data(start_time, end_time)

if df.empty:
    print("⚠️ Brak danych w podanym zakresie!")
    exit()

print(f"✅ Dane pobrane: {len(df)} rekordów\n")

# ===== 7️⃣ Przygotowanie danych =====

df = df.dropna()

X = df[['temperature', 'humidity', 'lux', 'charging_current']]
y = df['output_current']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===== 8️⃣ Trenowanie modelu =====

print("🚀 Rozpoczynam trenowanie modelu...")
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
print("✅ Model został przetrenowany.")

# ===== 9️⃣ Ewaluacja =====

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Wyniki trenowania ===")
print(f"🔍 Mean Squared Error (MSE): {mse:.4f}")
print(f"🔍 R2 Score: {r2:.4f}")

# ===== 🔄 10️⃣ Zapis modelu i scalera do Neon =====

print("\n💾 Zapisuję model i scaler do Neon...")

model_bytes = pickle.dumps(model)
scaler_bytes = pickle.dumps(scaler)

try:
    conn = get_neon_connection()
    cursor = conn.cursor()

    # Wstaw model (np. z nazwą 'model_v1')
    cursor.execute(
        "INSERT INTO models (name, model_data) VALUES (%s, %s)",
        ("model_v1", psycopg2.Binary(model_bytes))
    )

    # Wstaw scaler (np. z nazwą 'model_v1_scaler')
    cursor.execute(
        "INSERT INTO models (name, model_data) VALUES (%s, %s)",
        ("model_v1_scaler", psycopg2.Binary(scaler_bytes))
    )

    conn.commit()
    cursor.close()
    conn.close()

    print("✅ Model i scaler zostały zapisane w Neon (Postgres).")

except Exception as e:
    print(f"❌ Błąd zapisu do Neon: {e}")
