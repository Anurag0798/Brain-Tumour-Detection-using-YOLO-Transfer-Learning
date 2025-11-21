import requests

url = "http://127.0.0.1:8000/predict"

with open(r"dataset\images\train\00058_141.jpg", "rb") as f:
    response = requests.post(url = url, files = {"file": f})

print(response.json())