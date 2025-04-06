import requests
import streamlit as st

def login_user(username, password):
    response = requests.post(
        "https://iot-ai-backend.onrender.com/auth/token/jwt/create/",
        data={"username": username, "password": password}
    )
    
    if response.status_code == 200:
        token = response.json().get("access")
        st.session_state["token"] = token
        return True
    else:
        st.error("Błąd logowania: sprawdź dane")
        return False
