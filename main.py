import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router as routes_router
from metrics import router as metrics_router  # <— aggiungi

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_router)
app.include_router(metrics_router)  # <— aggiungi
print("ROUTER INCLUSO ✅")

