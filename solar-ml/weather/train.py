import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from influx.fetch_data import fetch_data

start_time = "2024-05-01T00:00:00Z"
end_time = "2024-05-02T00:00:00Z"

df = fetch_data(start_time, end_time)
df = df.dropna()

# Tworzymy etykiety: np. światło > 700 = słonecznie, < 300 = pochmurno
df['pogoda'] = df['swiatlo'].apply(lambda x: 'slonecznie' if x > 700 else ('pochmurno' if x < 300 else 'zmienne'))

X = df[["temp", "wilgotnosc", "swiatlo"]]
y = df["pogoda"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save model + scaler
joblib.dump(model, "model_weather.pkl")
joblib.dump(scaler, "scaler_weather.pkl")

print("✅ Model klasyfikacji zapisany jako model_weather.pkl")
