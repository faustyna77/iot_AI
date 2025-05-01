
from influxdb_client import InfluxDBClient
import os
from dotenv import load_dotenv
from collections import defaultdict
def get_data():
    url = os.getenv("INFLUXDB_URL")
    token = os.getenv("INFLUXDB_TOKEN")
    org = os.getenv("INFLUXDB_ORG")
    bucket = os.getenv("INFLUXDB_BUCKET")

    client = InfluxDBClient(url=url, token=token, org=org)

    query = f'''
    from(bucket: "{bucket}")
      |> range(start: -7h)
      |> filter(fn: (r) => r._measurement == "dht_measurements")
      |> filter(fn: (r) => r.device == "ESP32")
      |> filter(fn: (r) => r.location == "office")
      |> filter(fn: (r) => r.sensor == "DHT22")
    '''

    query_api = client.query_api()
    tables = query_api.query(query)

    data_by_time = defaultdict(dict)

    for table in tables:
        for record in table.records:
            timestamp = record.get_time().isoformat()
            field = record.get_field()
            value = record.get_value()
            data_by_time[timestamp][field] = value

    return dict(data_by_time)


if __name__ == "__main__":
    data = get_data()
    print(data)
