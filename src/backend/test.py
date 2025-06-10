import requests

API_URL = "https://api-inference.huggingface.co/models/Intelligent-Internet/II-Medical-8B"
headers = {
    "Authorization": f"Bearer YOUR_HUGGINGFACE_TOKEN"
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({"inputs": "What are the symptoms of diabetes?"})
print(output)
