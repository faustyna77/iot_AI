import psycopg2
import urllib.parse as up
import pickle
import os
import streamlit as st

def get_neon_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        st.error("❌ Nie znaleziono zmiennej środowiskowej DATABASE_URL!")
      
        
    up.uses_netloc.append("postgres")
    url = up.urlparse(DATABASE_URL)

    return psycopg2.connect(
        dbname=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port
    )

def load_model_and_scaler_from_neon(model_name):
    conn = get_neon_connection()
    cursor = conn.cursor()

    # Pobierz model
    cursor.execute("SELECT model_data FROM models WHERE name = %s ORDER BY created_at DESC LIMIT 1", (model_name,))
    model_row = cursor.fetchone()

    # Pobierz scaler
    cursor.execute("SELECT model_data FROM models WHERE name = %s ORDER BY created_at DESC LIMIT 1", (model_name + "_scaler",))
    scaler_row = cursor.fetchone()

    cursor.close()
    conn.close()

    if model_row and scaler_row:
        model = pickle.loads(model_row[0])
        scaler = pickle.loads(scaler_row[0])
        st.success(f"✅ Model '{model_name}' i scaler zostały wczytane z Neon.")
        return model, scaler
    else:
        st.success(f"❌ Nie znaleziono modelu lub scalera '{model_name}' w Neon.")
        return None, None


