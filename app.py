import streamlit as st
import sys
import os
from load_model import load_model_and_scaler_from_neon
import joblib
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from login import login_user
from register import register_user
import os
from dotenv import load_dotenv
import sys
from manual_control import write_decision
import psycopg2
import joblib
import streamlit as st
import numpy as np
import psycopg2
import urllib.parse as up
import pickle
import os


from agent import ai_decision

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
                    return result[['_time', 'temperature', 'humidity','lux','output_current','charging_current','predicted']]
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
            col1, col2,col3 = st.columns(3)
            col4,col5=st.columns(2)

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
            with col3:
                st.subheader("Oświetlenie")
                fig_lux = px.line(df_sensor, x='_time', y='lux', title="Natężenie oświetlenia")
                fig_lux.update_layout(yaxis_title="natężenie (lux)")
                st.plotly_chart(fig_lux, use_container_width=True)
            with col4:
                st.subheader("prąd ładowania")
                fig_charge = px.line(df_sensor, x='_time', y='charging_current', title="Prąd ładowania")
                fig_charge.update_layout(yaxis_title="prąd ładowania(A)")
                st.plotly_chart(fig_charge, use_container_width=True)
            with col5:
                st.subheader("prąd wyjściowy 1")
                fig_output = px.line(df_sensor, x='_time', y='output_current', title="Prąd wyjściowy 1")
                fig_output.update_layout(yaxis_title="prąd wyjściowy 1 (A)")
                st.plotly_chart(fig_output, use_container_width=True)
            # Tabela danych
            st.subheader("Ostatnie odczyty")
            st.dataframe(df_sensor.tail(800))
            st.title("📡 Monitorowanie systemu AI IoT")
           
        else:
            st.error("Brak dostępnych danych dla wybranego zakresu czasu.")
        
       
        model_option = st.selectbox(
                "Wybierz model AI:",
                (
                    "microsoft/mai-ds-r1:free",
                    "gpt-3.5-turbo",
                    "opengvlab/internvl3-14b:free",
                    "microsoft/mai-ds-r1:free"
                )
            )
        num_records = st.slider(
            "Ile ostatnich pomiarów przesłać do AI?",
            min_value=5,
            max_value=30,
            step=5,
            value=5
        )

        model_name=st.selectbox("Wybierz model do przewifdzenia:", ("model_v1", "model_v2"))

        model, scaler = load_model_and_scaler_from_neon(model_name)

        # Przewidywanie na podstawie ostatnich danych
        if st.button("🔮 Przewiduj prąd wyjściowy"):



            if not df_sensor.empty:
                


                last_row = df_sensor.iloc[-1]
                X_new = np.array([[last_row['temperature'], last_row['humidity'], last_row['lux'], last_row['charging_current']]])
                X_new_scaled = scaler.transform(X_new)
                y_pred = model.predict(X_new_scaled)
                st.success(f"🔮 Przewidywany prąd wyjściowy: {y_pred[0]:.2f} A")
            else:
                st.warning("Brak danych do przewidzenia!")

        if st.button("🔍 Zezwól na  wykonanie decyzji przez agenta "):

            st.session_state["run_agent"] = True

        if st.session_state.get("run_agent"):

            with st.spinner("Agent myśli..."):
                 
                 decision, reason = ai_decision(df_sensor, model_option, num_records)
                 st.success(f"🧠 Decyzja agenta: **{decision}**")
                 st.markdown(f"**Uzasadnienie:** {reason}")
            st.session_state["run_agent"] = False


             
                 
                 
                 
                 
                 
                   
                    
        if st.button("🔍 Sam wykonaj decyzję"):


            st.session_state["manual_mode"] = True

        if st.session_state.get("manual_mode"):
            st.markdown("### Wybierz decyzję ręcznie:")
            col1, col2, col3 = st.columns(3)
            if col1.button("⚡ CHARGE"):
                write_decision("CHARGE")
                st.session_state["manual_mode"] = False
            if col2.button("🔋 DISCHARGE"):
                write_decision("DISCHARGE")
                st.session_state["manual_mode"] = False
            if col3.button("⏸ OFF"):
                write_decision("OFF")
                st.session_state["manual_mode"] = False


        st.markdown("---")
        st.caption("Dane z InfluxDB & FATISSA'S TECHNOLOGY")
