from influxdb_client import InfluxDBClient, Point, WriteOptions
from import_data import get_data
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import streamlit as st
load_dotenv()

    # LLM model

sensor_data = get_data()


def ai_decision(sensor_data: dict, model_name: str, num_records: int) -> tuple[str, str]:
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=model_name,
        temperature=0.2,
    )
    
    formatted = "\n".join([f"{k}: {v}" for k, v in list(sensor_data.items())[-num_records:]])
   

    prompt = prompt = f"""
# Kontekst techniczny:
System fotowoltaiczny składa się z paneli 20W (maks. 26V), które ładują akumulator o pojemności 5Ah przez kontroler ładowania. 
Optymalny prąd ładowania akumulatora to 0.5A do 1.5A. 
Gdy prąd ładowania przekracza 1.5A przez dłuższy czas, może dojść do uszkodzenia akumulatora.
Gdy jest zbyt niski (<0.5A), ładowanie jest nieefektywne.

System mierzy:
- napięcie z paneli (`panel_voltage`)
- napięcie akumulatora (`battery_voltage`)
- prąd ładowania (`charging_current`)
- prąd rozładowania (`output_current`)
- temperaturę (`temperature`)
- wilgotność (`humidity`)
- nasłonecznienie (`lux`)

# Cel:
Na podstawie poniższych pomiarów podejmij decyzję:
- `CHARGE`: podłącz akumulator do paneli (jeśli warunki sprzyjają ładowaniu)
- `DISCHARGE`: pozwól akumulatorowi oddawać energię do odbiorników
- `OFF`: odłącz akumulator (np. gdy warunki nie sprzyjają ładowaniu lub rozładowanie jest ryzykowne)

# Ograniczenia:
- Nie ładuj, jeśli `charging_current` > 1.5A
- Nie ładuj, jeśli `charging_current` < 0.3A (mało efektywne)
- Nie rozładowuj, jeśli `battery_voltage` < 11.5V
- Preferuj ładowanie, gdy `lux` > 10000 i `panel_voltage` > 20V
- Preferuj wyłączenie (`OFF`) jeśli nasłonecznienie niskie, a rozładowanie duże

# Ostatnie {num_records} pomiarów:
{formatted}

Zalecana akcja (jedna z: CHARGE, DISCHARGE, OFF):
DECISION: ...
REASON: ...
"""


    response = llm.invoke([HumanMessage(content=prompt)])
    lines = response.content.splitlines()
    decision = ""
    reason = ""
    for line in lines:
        if line.upper().startswith("DECISION:"):
            decision = line.split(":")[1].strip()
        elif line.upper().startswith("REASON:"):
            reason = line.split(":")[1].strip()
    
    if decision:
        influx = InfluxDBClient(
            url=os.getenv("INFLUXDB_URL"),
            token=os.getenv("INFLUXDB_TOKEN"),
            org=os.getenv("INFLUXDB_ORG")
        )
        write_api = influx.write_api(write_options=WriteOptions(batch_size=1))
        
        point = Point("ai_decisions") \
            .field("decision", decision) \
            .field("reason", reason)\
            .time(datetime.now(timezone.utc))
            
        
        write_api.write(bucket=os.getenv("INFLUXDB_BUCKET"), record=point)
        st.success(f"✅ Zapisano decyzję: {decision}")

    else:
        st.warning("⚠️ Nie udało się odczytać decyzji z odpowiedzi.")

    return decision, reason






  
    
