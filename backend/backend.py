import requests

response = requests.get('http://web:8000/api_mazu')

data = response.json()

print(f"\nResponse from /api_mazu/:\n{data}\n")
print(type(data))
print(data.get('message'))