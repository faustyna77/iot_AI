import pandas as pd
import streamlit as st
from influxdb_client import InfluxDBClient
from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET

# Inicjalizacja klienta InfluxDB
try:
    influx_client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )
except Exception as e:
    print(f"❌ Błąd podczas inicjalizacji klienta InfluxDB: {e}")
   

# Funkcja do zapytania o dane z sensora
def query_sensor_data(start_time, end_time):
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {start_time}, stop: {end_time})
        |> filter(fn: (r) => r["_measurement"] == "dht_measurements")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        result = influx_client.query_api().query_data_frame(query)
        if not result.empty:
            result['_time'] = pd.to_datetime(result['_time'])
            return result[['_time', 'temperature', 'humidity', 'lux', 'output_current', 'charging_current', 'predicted']]
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Błąd zapytania sensorów: {e}")
        return pd.DataFrame()
