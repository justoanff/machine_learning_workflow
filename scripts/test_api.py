import requests

url = "http://localhost:8080/predict"
data = {"texts": ["This is a positive review", "This movie was terrible"]}
response = requests.post(url, json=data)
print(response.json())

