from influxdb_client import InfluxDBClient, Point, WriteOptions
from datetime import datetime, timezone
import os
import streamlit as st

def write_decision(decision):
   

    client = InfluxDBClient(
        url=st.secrets["INFLUXDB_URL"],
        token=st.secrets["INFLUXDB_TOKEN"],
        org=st.secrets["INFLUXDB_ORG"]
    )
    write_api = client.write_api(write_options=WriteOptions(batch_size=1))
    
    point = Point("ai_decisions") \
        .field("decision", decision) \
        .field("reason", "manual override by user") \
        .time(datetime.now(timezone.utc))
        

    try:
        write_api.write(bucket=st.secrets["INFLUXDB_BUCKET"], record=point)
        st.success("✅ Udało się zapisać punkt!")
    except Exception as e:
        st.error("❌ Błąd przy zapisie:", e)




