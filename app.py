import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from login import login_user
from register import register_user
import os
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja strony
st.set_page_config(page_title="Faustyna Misiura", layout="wide")
st.title("Integracja cyberbezpieczeństwa i sztucznej inteligencji w predykcyjnym sterowaniu systemami IoT infrastruktury krytycznej - Wyniki badań")
query_params = st.query_params

if "token" not in st.session_state:
    if "token" in query_params:
        st.session_state["token"] = query_params["token"]
        st.success("✅ Zalogowano przez Google!")
page = st.sidebar.radio("🔐 Nawigacja", ["Logowanie", "Rejestracja"])


google_login_url = "https://iot-ai-backend.onrender.com/auth/login/google-oauth2/?next=https://fastinatechnology.streamlit.app"



if page=="Rejestracja":
    register_user()
else:

    
    # Interfejs logowania
    if "token" not in st.session_state:
        st.subheader("🔐 Zaloguj się")
        username = st.text_input("Login")
        password = st.text_input("Hasło", type="password")

        if st.button("Zaloguj"):
            if login_user(username, password):
                st.success("✅ Zalogowano pomyślnie!")
                st.rerun()
        st.markdown("### 🌐 Albo zaloguj się przez Google")
        google_login_url = "https://iot-ai-backend.onrender.com/auth/login/google-oauth2/"
        st.markdown(f"[Zaloguj się przez Google]({google_login_url})", unsafe_allow_html=True)

    # Po zalogowaniu
    if "token" in st.session_state:

        # Pobieranie zmiennych środowiskowych
        
        load_dotenv()  # Załaduj zmienne z pliku .env

        INFLUXDB_URL = st.secrets["INFLUXDB_URL"]
        INFLUXDB_TOKEN = st.secrets["INFLUXDB_TOKEN"]
        INFLUXDB_ORG = st.secrets["INFLUXDB_ORG"]
        INFLUXDB_BUCKET = st.secrets["INFLUXDB_BUCKET"]
        print("INFLUXDB_URL:", st.secrets("INFLUXDB_URL"))
        print("INFLUXDB_TOKEN:", st.secrets("INFLUXDB_TOKEN"))
        print("INFLUXDB_ORG:", st.secrets("INFLUXDB_ORG"))
        print("INFLUXDB_BUCKET:", st.secrets("INFLUXDB_BUCKET"))


        THRESHOLD = 21.47  # Próg temperatury

        if not all([INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET]):
            st.error("Nie wszystkie zmienne środowiskowe zostały załadowane!")
            st.stop()

        # Inicjalizacja klienta InfluxDB
        try:
            client = InfluxDBClient(
                url=INFLUXDB_URL,
                token=INFLUXDB_TOKEN,
                org=INFLUXDB_ORG
            )
        except Exception as e:
            st.error(f"Błąd podczas inicjalizacji klienta InfluxDB: {e}")
            st.stop()

        # Funkcja do zapytania o dane z sensora
        def query_sensor_data(start_time, end_time):
            query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {start_time}, stop: {end_time})
                |> filter(fn: (r) => r["_measurement"] == "dht_measurements")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            try:
                result = client.query_api().query_data_frame(query)
                if not result.empty:
                    result['_time'] = pd.to_datetime(result['_time'])
                    return result[['_time', 'temperature', 'humidity']]
                return pd.DataFrame()
            except Exception as e:
                st.error(f"Błąd zapytania sensorów: {e}")
                return pd.DataFrame()

        # Funkcja do zapytania o przewidywaną temperaturę
        def query_predicted_temp(start_time, end_time):
            query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {start_time}, stop: {end_time})
                |> filter(fn: (r) => r._measurement == "predicted_temperature")
                |> filter(fn: (r) => r._field == "value")
            '''
            try:
                result = client.query_api().query_data_frame(query)
                if not result.empty:
                    result['_time'] = pd.to_datetime(result['_time'])
                    return result[['_time', '_value']].rename(columns={'_value': 'predicted_temperature'})
                return pd.DataFrame()
            except Exception as e:
                st.error(f"Błąd zapytania przewidywań: {e}")
                return pd.DataFrame()

        # Panel boczny z wyborem zakresu czasu
        st.sidebar.header("Filtry")
        time_range = st.sidebar.selectbox(
            "Wybierz zakres czasu",
            ["godzina", "24 poprzednie", "zeszły tydzień", "zeszły miesiąc"]
        )

        # Obliczanie zakresu czasu
        now = datetime.utcnow()
        if time_range == "godzina":
            start_time = now - timedelta(hours=1)
        elif time_range == "24 poprzednie":
            start_time = now - timedelta(days=1)
        elif time_range == "zeszły tydzień":
            start_time = now - timedelta(weeks=1)
        else:
            start_time = now - timedelta(days=30)

        end_time = now

        # Pobieranie danych
        df_sensor = query_sensor_data(start_time.isoformat() + "Z", end_time.isoformat() + "Z")
        df_predicted = query_predicted_temp(start_time.isoformat() + "Z", end_time.isoformat() + "Z")

        # Sprawdzanie, czy dane są dostępne
        if not df_sensor.empty:
            col1, col2 = st.columns(2)

            # Status systemu awaryjnego
            if not df_predicted.empty:
                latest_predicted = df_predicted['predicted_temperature'].iloc[-1]
                led_status = "🔴 ON" if latest_predicted > THRESHOLD else "⚪ OFF"
                st.info(f"""
                ### Status systemu awaryjnego
                - LED: {led_status}
                - Przewidywana temperatura: {latest_predicted:.2f}°C
                - Próg: {THRESHOLD:.2f}°C
                """)

            # Wykres temperatury
            with col1:
                st.subheader("Temperatura")
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(x=df_sensor['_time'], y=df_sensor['temperature'], name='Temperatura', line=dict(color='blue')))
                if not df_predicted.empty:
                    fig_temp.add_trace(go.Scatter(x=df_predicted['_time'], y=df_predicted['predicted_temperature'], name='Przewidywana', line=dict(color='red', dash='dash')))
                fig_temp.add_hline(y=THRESHOLD, line_dash="dot", line_color="green", annotation_text=f"Próg: {THRESHOLD}°C")
                st.plotly_chart(fig_temp, use_container_width=True)

            # Wykres wilgotności
            with col2:
                st.subheader("Wilgotność")
                fig_humid = px.line(df_sensor, x='_time', y='humidity', title="Wilgotność")
                fig_humid.update_layout(yaxis_title="Wilgotność (%)")
                st.plotly_chart(fig_humid, use_container_width=True)

            # Tabela danych
            st.subheader("Ostatnie odczyty")
            st.dataframe(df_sensor.tail(10))
        else:
            st.error("Brak dostępnych danych dla wybranego zakresu czasu.")

        # Stopka
        st.markdown("---")
        st.caption("Dane z InfluxDB & FATISSA'S TECHNOLOGY")
