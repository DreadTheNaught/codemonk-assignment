# api test script

import requests

with open("images/fashion_image.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post("http://localhost:5000/predict", files=files)

print(response.json())
