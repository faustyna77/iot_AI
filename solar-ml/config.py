import os
from dotenv import load_dotenv

# Ładowanie pliku .env (domyślnie z katalogu głównego projektu)
load_dotenv()

# Czytamy zmienne z .env
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
