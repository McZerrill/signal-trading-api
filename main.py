from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router as binance_router      # router Binance (quello che già usi)
from yahoo_routes import router as yahoo_router  # nuovo router Yahoo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Endpoints Binance: /analyze, /price, /hotassets, /simulazioni_attive, ecc.
app.include_router(binance_router)

# 🔹 Endpoints Yahoo: /analyze_yahoo, /price_yahoo
app.include_router(yahoo_router)

print("ROUTER INCLUSI ✅ (Binance + Yahoo)")
