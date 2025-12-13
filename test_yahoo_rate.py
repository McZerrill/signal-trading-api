import time
import requests

BASE = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"  # oro future
PARAMS = {"interval": "15m", "range": "7d"}

def do_call(i):
    r = requests.get(BASE, params=PARAMS, timeout=5)
    print(
        f"{time.strftime('%H:%M:%S')}  call #{i}  status={r.status_code}  len={len(r.content)}"
    )

if __name__ == "__main__":
    for i in range(1, 51):  # prova 50 chiamate
        do_call(i)
        time.sleep(2.0)     # cambia a 1.0 / 3.0 / 5.0 per testare diversi ritmi
