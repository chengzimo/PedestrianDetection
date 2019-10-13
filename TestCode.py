#encoding: utf-8
import base64
import requests

url = 'http://localhost:8086/PedestrianDetection'

with open(u'Person.jpg', "rb") as f:
    img = f.read()
base64Data = base64.b64encode(img)

data = {'ImageData': base64Data}
response = requests.post(url, data=data)
print(response.text)