import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Słownik: cechy wejściowe wymagane przez każdy model
model_features = {
    "model_v1": {
        "label": "output_current",
        "features": ["temperature", "humidity", "lux", "charging_current"]
    },
     "model_v1_anomalion_off": {
        "label": "output_current",
        "features": ["temperature", "humidity", "lux", "charging_current"]
    },

    
    "model_lux": {
        "label": "lux",
        "features": ["temperature", "humidity", "charging_current", "output_current"]
    },
     "model_lux_anomalion_off": {
        "label": "lux",
        "features": ["temperature", "humidity", "charging_current", "output_current"]
    },
    "model_temperature": {
        "label": "temperature",
        "features": ["humidity", "lux", "output_current", "charging_current"]
    },
     "model_temperature_anomalion_off": {
        "label": "temperature",
        "features": ["humidity", "lux", "output_current", "charging_current"]
    },
    
     "model_humidity": {
        "label": "humidity",
        "features": ["temperature", "lux", "output_current", "charging_current"]
    },
     "model_humidity_anomalion_off": {
        "label": "humidity",
        "features": ["temperature", "lux", "output_current", "charging_current"]
    },

   
    "model_charging": {
        "label": "charging_current",
        "features": ["temperature", "humidity", "lux", "output_current"]
    },
     "model_charging_anomalion_off": {
        "label": "charging_current",
        "features": ["temperature", "humidity", "lux", "output_current"]
    },


   
}



from sklearn.ensemble import IsolationForest

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

    # 🔍 Filtrowanie anomalii tylko dla określonego modelu
    if model_name == "model_v1_anomalion_off":
        clf = IsolationForest(contamination=0.05, random_state=42)
        mask = clf.fit_predict(df_sensor[required_features]) == 1
        df_sensor = df_sensor[mask]

    try:
        X_input = df_sensor[required_features]
        X_scaled = scaler.transform(X_input)
        y_pred = model.predict(X_scaled)

        df_sensor["prediction"] = y_pred

        st.subheader(f"📈 Predykcja {label} w czasie")
        fig = px.line(df_sensor, x="_time", y="prediction", title=f"Predykcja {label} w czasie")
        fig.update_layout(yaxis_title=label)
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"🔮 Średnia przewidywana wartość ({label}): {np.mean(abs(y_pred)):.2f}")
        st.info(f"""
            📊 Statystyki predykcji:
            - Średnia: {np.mean(abs(y_pred)):.2f}
            - Mediana: {np.median(abs(y_pred)):.2f}
            - Min: {np.min(abs(y_pred)):.2f}
            - Max: {np.max(abs(y_pred)):.2f}
        """)

    except KeyError as e:
        st.error(f"❌ Brakuje danych wejściowych: {e}")
    except Exception as e:
        st.error(f"❌ Błąd podczas predykcji: {e}")
