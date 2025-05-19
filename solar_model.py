import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import plotly.express as px
from analysis import show_correlation, show_anomalies

# Słownik: cechy wejściowe wymagane przez każdy model
model_features = {
    "model_v1": {
        "label": "prąd wyjściowy",
        "features": ["temperature", "humidity", "lux", "charging_current"]
    },
    "model_lux": {
        "label": "oświetlenie (lux)",
        "features": ["temperature", "humidity", "charging_current", "output_current"]
    },
    "model_temperature": {
        "label": "temperatura",
        "features": ["humidity", "lux", "output_current", "charging_current"]
    },
    "model_humidity": {
        "label": "humidity",
        "features": ["temperature", "lux", "output_current", "charging_current"]
    },
    "model_charging": {
        "label": "charging_current",
        "features": ["temperature", "humidity", "lux", "output_current"]
    },
}

def predict(model, scaler, df_sensor, model_name):
    if model_name not in model_features:
        st.error(f"⚠️ Model '{model_name}' nie jest skonfigurowany.")
        return

    config = model_features[model_name]
    label = config["label"]
    required_features = config["features"]

    if df_sensor.empty:
        st.warning("Brak danych do przewidzenia!")
        return

    try:
        # Przygotuj dane dla wszystkich wierszy
        X_input = df_sensor[required_features]
        X_scaled = scaler.transform(X_input)
        y_pred = model.predict(X_scaled)

        # Wykres predykcji vs czas
        st.subheader(f"🌇 Predykcja {label} dla wybranego okresu")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_sensor['_time'], y_pred, label='Predykcja', color='orange')
        ax.set_xlabel("Czas")
        ax.set_ylabel(label)
        ax.set_title(f"Predykcja {label} w czasie")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Pokaż ostatni wynik predykcji
        st.success(f"🔮 Ostatnia przewidywana wartość ({label}): {y_pred[-1]:.2f}")

        # Korelacja
        show_correlation(df_sensor)

        # Detekcja anomalii
        show_anomalies(df_sensor)

    except KeyError as e:
        st.error(f"❌ Brakuje danych wejściowych: {e}")
    except Exception as e:
        st.error(f"❌ Błąd podczas predykcji: {e}")
