from influxdb_client import InfluxDBClient, Point, WriteOptions
from import_data import get_data
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()

    # LLM model
llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="microsoft/mai-ds-r1:free",
        temperature=0.2,
    )
sensor_data = get_data()


def ai_decision(sensor_data: dict) -> tuple[str, str]:
    
    formatted = "\n".join([f"{k}: {v}" for k, v in list(sensor_data.items())[-5:]])

    prompt = f"""
Dane z czujników:
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
        print("✅ Zapisano decyzję:", decision)
    else:
        print("⚠️ Nie udało się odczytać decyzji z odpowiedzi.")

    return decision, reason






  
    