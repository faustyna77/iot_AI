from influxdb_client import InfluxDBClient, Point, WriteOptions
from datetime import datetime, timezone
import os

def write_decision(decision):
   

    client = InfluxDBClient(
        url=os.getenv("INFLUXDB_URL"),
        token=os.getenv("INFLUXDB_TOKEN"),
        org=os.getenv("INFLUXDB_ORG")
    )
    write_api = client.write_api(write_options=WriteOptions(batch_size=1))
    
    point = Point("ai_decisions") \
        .field("decision", decision) \
        .field("reason", "manual override by user") \
        .time(datetime.now(timezone.utc))
        

    try:
        write_api.write(bucket=os.getenv("INFLUXDB_BUCKET"), record=point)
        print("✅ Udało się zapisać punkt!")
    except Exception as e:
        print("❌ Błąd przy zapisie:", e)




