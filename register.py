import streamlit as st
import requests

def register_user():
    st.subheader("📝 Zarejestruj się")

    username = st.text_input("Nazwa użytkownika", key="reg_username")
    password = st.text_input("Hasło", type="password", key="reg_password")
    re_password = st.text_input("Powtórz hasło", type="password", key="reg_repassword")

    if st.button("Zarejestruj się"):
        if password != re_password:
            st.error("❗ Hasła się nie zgadzają.")
        elif not username or not password:
            st.warning("❗ Wszystkie pola są wymagane.")
        else:
            try:
                response = requests.post(
                    "https://iot-ai-backend.onrender.com/auth/core/users/",
                    json={
                        "username": username,
                        "password": password,
                        "re_password": re_password
                    },
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 201:
                    st.success("✅ Konto utworzone! Możesz się teraz zalogować.")
                else:
                    st.error(f"❌ Błąd: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"🚨 Błąd połączenia z backendem: {e}")
