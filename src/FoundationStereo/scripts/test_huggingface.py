# filepath: scripts/test_huggingface.py
import requests

proxies = {
    "http": "http://127.0.0.1:33057",
    "https": "http://127.0.0.1:33057",
}
try:
    r = requests.get("https://huggingface.co", proxies=proxies, timeout=10)
    print("Status:", r.status_code)
except Exception as e:
    print("Error:", e)
