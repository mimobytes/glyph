import requests

url = "http://127.0.0.1:5000/analyze"
file_path = "k.png"

with open(file_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

print("status:", response.status_code)
print("json:", response.json())
