import requests
import csv
import pandas as pd

url = "http://122.155.202.69:1180/cu_amd/api/get_log.php"

querystring = {"log_type":"accelerometer_log","offset":"0","range":"500","user_id":"all"}

payload = ""
headers = {
    'Authorization': "admin",
    'cache-control': "no-cache",
    'Postman-Token': "a40e25c3-dfc6-4a9f-977f-227effeafe0b"
    }

res = requests.request("GET", url, data=payload, headers=headers, params=querystring)
json = res.json()

df = pd.DataFrame({
		'timestamp': [x['event_timestamp'] for x in json],
		'x': [x['y'] for x in json],
		'y': [x['y'] for x in json],
		'z': [x['z'] for x in json]
	})

df['index'] = df.index

columns = ['timestamp','x','y','z']

df = df.sort_values(by='index',ascending=False)
df = df.reset_index()
df = df[columns]

print(df)

df.to_csv('SmartwatchData/acc_db_1.csv')