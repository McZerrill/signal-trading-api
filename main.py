from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router as binance_router
from routes import start_background_tasks
# from yahoo_routes import router as yahoo_router  # legacy


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(binance_router)
# app.include_router(yahoo_router)  # ⛔ legacy, Yahoo ora è gestito in routes.py

@app.on_event("startup")
def _startup():
    start_background_tasks()

print("ROUTER INCLUSI ✅ (Backend unificato)")

