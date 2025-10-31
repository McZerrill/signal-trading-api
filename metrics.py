# metrics.py
from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

router = APIRouter()

# Metriche Prometheus
REQS = Counter("seg_analizza_requests_total", "Totale richieste a /analyze")
ERRS = Counter("seg_analizza_errors_total", "Errori in /analyze")
LAT  = Histogram(
    "seg_analizza_latency_seconds",
    "Latenza /analyze",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5]
)
SIMS = Gauge("seg_simulazioni_attive", "Simulazioni attive")

@router.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type="text/plain")

# ---- Utility leggere per non riscrivere ovunque lo stesso codice ----

def timer_start() -> float:
    return time.perf_counter()

def timer_observe(t0: float) -> None:
    LAT.observe(time.perf_counter() - t0)

def inc_request() -> None:
    REQS.inc()

def inc_error() -> None:
    ERRS.inc()

def set_sim_count(n: int) -> None:
    try:
        SIMS.set(n)
    except Exception:
        # non bloccare il flusso applicativo per telemetria
        pass
