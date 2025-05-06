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
        api_key=st.secrets["OPENROUTER_API_KEY"],
        model=model_name,
        temperature=0.2,
    )
    
    formatted = "\n".join([f"{k}: {v}" for k, v in list(sensor_data.items())[-num_records:]])
   

    prompt = f"""
Dane z czujników (ostatnie {num_records} pomiarów):
{formatted}

Zalecana akcja (jedna z: CHARGE, DISCHARGE, OFF).
Format:
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
            url=st.secrets["INFLUXDB_URL"],
            token=st.secrets["INFLUXDB_TOKEN"],
            org=st.secrets["INFLUXDB_ORG"]
        )
        write_api = influx.write_api(write_options=WriteOptions(batch_size=1))
        
        point = Point("ai_decisions") \
            .field("decision", decision) \
            .field("reason", reason)\
            .time(datetime.now(timezone.utc))
            
        
        write_api.write(bucket=st.secrets["INFLUXDB_BUCKET"], record=point)
        st.success(f"✅ Zapisano decyzję: **{decision}**")

    else:
        st.error("⚠️ Nie udało się odczytać decyzji z odpowiedzi.")

    return decision, reason






  
    