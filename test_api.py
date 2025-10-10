import requests

url = "http://127.0.0.1:8080/recommend"
data = {"user_id": 1, "k": 10}
resp = requests.post(url, json=data)
print(resp.json())
