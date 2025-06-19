import streamlit as st
import sys
import os
from train_charging_anomalion_off import train_charging_anomalion_off
from analysis import show_anomalies, show_correlation, show_cleaned_series
from train_lux_anomalion_off import train_lux_anomalion_off
from load_model import load_model_and_scaler_from_neon
from prediction import predict, model_features
from train_lux import train_lux
from train_temperature import train_temperature
from train_humidity import train_humidity
from train_charging_current import train_charging_current
from train import train
from train_humidity_anomalion_off import train_humidity_anomalion_off
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
from train_isfores import train_isolation
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings  # lub HuggingFaceEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

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
        DATABASE_URL = st.secrets["DATABASE_URL"] 
        



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
                |> filter(fn: (r) => r["_measurement"] == "env_measurements")
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
        def query_voltage_data(start_time, end_time):
            query_vol = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {start_time}, stop: {end_time})
                |> filter(fn: (r) => r["_measurement"] == "vol_measurements")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            try:
                result_vol = client.query_api().query_data_frame(query_vol)
                if not result_vol.empty:
                    result_vol['_time'] = pd.to_datetime(result_vol['_time'])
                    return result_vol[['_time', 'acu_power', 'acu_voltage','battery_percent','panel_power','panel_voltage']]
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
        now =datetime.utcnow()
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
        df_voltage=query_voltage_data(start_time.isoformat() + "Z", end_time.isoformat() + "Z")
        df_predicted = query_predicted_temp(start_time.isoformat() + "Z", end_time.isoformat() + "Z")
        

        # Sprawdzanie, czy dane są dostępne
        if not df_sensor.empty:
            col1, col2,col3 = st.columns(3)
            col4,col5=st.columns(2)
            col6,col7,col8=st.columns(3)

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
            with col6:
                st.subheader("Napięcie akumulatora")
                fig_acu_vol = px.line(df_voltage, x='_time', y='acu_voltage', title="Napięcie [V]")
                fig_acu_vol.update_layout(yaxis_title="napięcie akumulatora [V]")
                st.plotly_chart(fig_acu_vol, use_container_width=True)
            with col7:
                st.subheader("Procent naładowania akumulatora")
                fig_acu_percent = px.line(df_voltage, x='_time', y='battery_percent', title="Procent[%]")
                fig_acu_percent.update_layout(yaxis_title="procent naładowania")
                st.plotly_chart(fig_acu_percent, use_container_width=True)
            with col8:
                st.subheader("Napięcie paneli")
                fig_panel_vol = px.line(df_voltage, x='_time', y='panel_voltage', title="Napięcie paneli [V]")
                fig_panel_vol.update_layout(yaxis_title="napiecie wyjściowe paneli")
                st.plotly_chart(fig_panel_vol, use_container_width=True)
            
            # Tabela danych
            st.subheader("Ostatnie odczyty")
            st.dataframe(df_sensor.tail(800))
            st.title("📡 Monitorowanie systemu AI IoT")
            if "show_training" not in st.session_state:
                    
                    st.session_state["show_training"] = False

            if st.button("🧠 Przetrenuj modele"):
                    st.session_state["show_training"] = not st.session_state["show_training"]

            if st.session_state["show_training"]:
                    st.subheader("⚙️ Wybierz model do trenowania")

                    if st.button("🔁 Trenuj model_lux"):
                        train_lux()
                    if st.button("🌡️ Trenuj model_temperature"):
                        train_temperature()
                    if st.button("💧 Trenuj model_humidity"):
                        train_humidity()
                    if st.button("🔌 Trenuj model_charging"):
                        train_charging_current()
                    if st.button("⚡ Trenuj model_v1 (output_current)"):
                        train()

                    if st.button("⚡ Trenuj model_v1_ anomalion off (output_current)"):
                        train_isolation()
                    if st.button("⚡ Trenuj model_lux_ anomalion off (lux)"):
                        train_lux_anomalion_off()
                    if st.button("Trenuj model_humidity_ anomalion off "):
                        train_humidity_anomalion_off()
                    if st.button("Trenuj model_charging_current_anomalion off ⚡"):
                        train_charging_anomalion_off()
                    


                        
           
        else:
            st.error("Brak dostępnych danych dla wybranego zakresu czasu.")
        
        
       
        model_name = st.selectbox("Wybierz model do przewidzenia:", list(model_features.keys()))
        model, scaler = load_model_and_scaler_from_neon(model_name)
        # Przewidywanie na podstawie ostatnich danych
        # --- Przycisk do przewidywania ---
        if st.button("🔮 Przewiduj"):

            from analysis import show_anomalies

            predict(model, scaler, df_sensor, model_name)

    # Automatycznie pobierz nazwę zmiennej do analizy anomalii
            label = model_features[model_name]["label"]
            show_anomalies(df_sensor, label)
            show_cleaned_series(df_sensor, label=model_features[model_name]["label"])
            
            
        # --- Obsługa trybu agentów AI ---
        if "run_session" not in st.session_state:
            st.session_state["run_session"] = False
        if "run_agent" not in st.session_state:
            st.session_state["run_agent"] = False

        # --- Pokazanie panelu agenta AI ---
        if st.button("🔄 Sterowanie za pomocą agentów AI"):
            st.session_state["run_session"] = True

        if st.session_state["run_session"]:
            model_option = st.selectbox(
                "Wybierz model AI:",
                (
                    "microsoft/mai-ds-r1:free",
                    "gpt-3.5-turbo",
                    "opengvlab/internvl3-14b:free",
                    "moonshotai/kimi-dev-72b:free",
                    "deepseek/deepseek-r1-0528-qwen3-8b:free",
                    "sarvamai/sarvam-m:free",
                    "mistralai/devstral-small:free",
                    "google/gemma-3n-e4b-it:free",
                    "meta-llama/llama-3.3-8b-instruct:free",
                    "nousresearch/deephermes-3-mistral-24b-preview:free",
                    "qwen/qwen3-30b-a3b:free",
                    "tngtech/deepseek-r1t-chimera:free",
                    "shisa-ai/shisa-v2-llama3.3-70b:free",
                    "agentica-org/deepcoder-14b-preview:free"

                )
            )
            num_records = st.slider(
                "Ile ostatnich pomiarów przesłać do agenta AI?",
                min_value=5,
                max_value=30,
                step=5,
                value=5
            )

            if st.button("🔍 Zezwól na wykonanie decyzji przez agenta"):
                with st.spinner("Agent myśli..."):
                    decision, reason = ai_decision(df_sensor, model_option, num_records)
                    st.success(f"🧠 Decyzja agenta: **{decision}**")
                    st.markdown(f"**Uzasadnienie:** {reason}")

                


             
                 
                 
                 
                 
                 
                   
                    
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
        openrouter_key = st.secrets["OPENROUTER_API_KEY"]
                # 1. Wczytaj PDF i przygotuj chunki
        @st.cache_resource(show_spinner="🔄 Ładowanie dokumentu...")
        def load_and_split_pdf(pdf_path):
            reader = PdfReader(pdf_path)
            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            return splitter.split_text(raw_text)

        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, "Integracja_v9.2.pdf")
        chunks = load_and_split_pdf(pdf_path)


        # 2. Embeddingi i wektoryzacja przez FAISS
        @st.cache_resource(show_spinner="🔎 Generowanie wektorów...")
        def create_qa_chain(chunks):
            # Użyj lokalnego modelu do embeddingów
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # FAISS wektorowa baza
            vectorstore = FAISS.from_texts(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # OpenRouter LLM
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model="openai/gpt-4",  # lub inny np. "mistralai/mistral-7b-instruct"
                temperature=0.2,
            )

            return RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever
            )

        qa_chain = create_qa_chain(chunks)

        # 3. Interfejs czatu
        st.title("📘 Chat z pracą magisterską (RAG + FAISS)")
        st.markdown("Zadaj pytanie dotyczące treści pracy :")

        user_question = st.text_input("✍️ Twoje pytanie:")

        if user_question:
            with st.spinner("🤔 Szukam odpowiedzi..."):
                response = qa_chain.run(user_question)
                st.success(response)
        st.caption("Dane z InfluxDB & FATISSA'S TECHNOLOGY")
