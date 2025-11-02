# main.py
import os, sys, json, threading, time, logging
from typing import Any, Dict
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importa router e riferimenti allo stato
from routes import router as routes_router
from metrics import router as metrics_router

# ⬇️ importa lo stato condiviso e (opzionale) l’avvio del monitor se lo esponi
try:
    from routes import posizioni_attive  # dict condiviso
except Exception:
    posizioni_attive = {}  # fallback, ma è meglio importare quello reale

try:
    from routes import start_monitor_thread  # funzione opzionale
except Exception:
    start_monitor_thread = None  # non disponibile, ignora

AUTOSAVE_PATH = os.getenv("POS_SAVE_PATH", "posizioni_attive.json")
AUTOSAVE_SEC  = int(os.getenv("POS_AUTOSAVE_SEC", "10"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_router)
app.include_router(metrics_router)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("ROUTER INCLUSO ✅")

# -----------------------------------------------------------------------------
# Autosave loop
# -----------------------------------------------------------------------------
_stop_event = threading.Event()

def _load_state_from_disk(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path) and os.path.getsize(path) > 2:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                logging.info(f"🔁 Stato ripristinato da {path} ({len(data)} simulazioni)")
                return data
    except Exception as e:
        logging.warning(f"⚠️ Ripristino stato fallito: {e}")
    return {}

def _autosave_loop():
    while not _stop_event.is_set():
        try:
            # snapshot atomico semplice
            snap = dict(posizioni_attive) if isinstance(posizioni_attive, dict) else {}
            with open(AUTOSAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"autosave error: {e}")
        _stop_event.wait(AUTOSAVE_SEC)

# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _on_startup():
    # 1) ripristino stato
    try:
        restored = _load_state_from_disk(AUTOSAVE_PATH)
        if restored:
            # aggiorna in place per non rompere eventuali riferimenti
            posizioni_attive.clear()
            posizioni_attive.update(restored)
    except Exception as e:
        logging.warning(f"⚠️ Problema durante il ripristino stato: {e}")

    # 2) autosave
    t = threading.Thread(target=_autosave_loop, name="autosave", daemon=True)
    t.start()
    logging.info(f"💾 Autosave attivo ogni {AUTOSAVE_SEC}s → {AUTOSAVE_PATH}")

    # 3) monitor backend (se esposto dal router)
    if callable(start_monitor_thread):
        try:
            start_monitor_thread()
            logging.info("🛰️ Monitor thread avviato dal main")
        except Exception as e:
            logging.exception(f"❌ Avvio monitor fallito: {e}")
    else:
        logging.info("ℹ️ start_monitor_thread() non esposto: il monitor parte da routes.py")

@app.on_event("shutdown")
def _on_shutdown():
    _stop_event.set()
    # best-effort save
    try:
        with open(AUTOSAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(posizioni_attive), f, ensure_ascii=False)
        logging.info("💾 Stato salvato in shutdown")
    except Exception as e:
        logging.warning(f"⚠️ Salvataggio in shutdown fallito: {e}")

# -----------------------------------------------------------------------------
# Healthcheck minimale
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    try:
        n = len(posizioni_attive) if isinstance(posizioni_attive, dict) else 0
    except Exception:
        n = -1
    return {"ok": True, "simulazioni": n}

