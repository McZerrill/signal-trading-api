from fastapi import FastAPI
from binance.client import Client
from dotenv import load_dotenv
import os

app = FastAPI()  # <--- questa riga è fondamentale!

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

client = Client(api_key, api_secret)

@app.get("/test-binance")
def test_binance():
    try:
        server_time = client.get_server_time()
        return {"success": True, "server_time": server_time}
    except Exception as e:
        return {"success": False, "error": str(e)}
