# capital_scaler.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from threading import RLock
from time import time
from typing import Dict, Optional, Tuple


@dataclass
class ScaleState:
    step_index: int = 0          # 0..3 (4 step totali) -> dopo start diventa 1
    last_ref_price: float = 0.0  # prezzo di riferimento ultimo step
    rebuy_ready: bool = False    # True quando il prezzo scende "invece di salire"
    last_update_ts: float = field(default_factory=lambda: time())


class CapitalScaler:
    """
    Scalatura 30%-20%-20%-30% su BUY successivi, avanzando SOLO se:
    - la simulazione è stata avviata (on_simulation_started)
    - dopo l'ultimo step il prezzo è sceso abbastanza (rebuy_ready)

    Persistenza su file JSON: salva SOLO gli asset "veri" avviati in simulazione.
    """

    def __init__(
        self,
        tranches_pct: Tuple[int, int, int, int] = (30, 20, 20, 30),
        dd_trigger_pct: float = 0.008,       # -0.8% abilita rebuy_ready
        reset_profit_pct: float = 0.012,     # +1.2% resetta ciclo (ha preso direzione)
        ttl_seconds: int = 6 * 60 * 60,      # stato valido 6 ore
        state_file: str = "data/capital_scaler_state.json",
    ):
        self.tranches_pct = tranches_pct
        self.dd_trigger_pct = dd_trigger_pct
        self.reset_profit_pct = reset_profit_pct
        self.ttl_seconds = ttl_seconds
        self.state_file = state_file

        self._lock = RLock()
        self._states: Dict[str, ScaleState] = {}

        self._load_from_disk()

    # -------------------------
    # Persistenza (JSON)
    # -------------------------
    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.state_file)
        if d:
            os.makedirs(d, exist_ok=True)

    def _save_to_disk(self) -> None:
        """
        Salvataggio atomico: scrive su .tmp e poi os.replace.
        """
        self._ensure_dir()
        tmp = self.state_file + ".tmp"

        payload = {
            "version": 1,
            "saved_at": time(),
            "states": {sym: asdict(st) for sym, st in self._states.items()},
        }

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        os.replace(tmp, self.state_file)

    def _load_from_disk(self) -> None:
        """
        Carica stati persistiti (se esistono).
        """
        try:
            if not os.path.exists(self.state_file):
                return
            with open(self.state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            states = payload.get("states", {})
            loaded: Dict[str, ScaleState] = {}
            now = time()

            for sym, st_dict in states.items():
                try:
                    st = ScaleState(
                        step_index=int(st_dict.get("step_index", 0)),
                        last_ref_price=float(st_dict.get("last_ref_price", 0.0)),
                        rebuy_ready=bool(st_dict.get("rebuy_ready", False)),
                        last_update_ts=float(st_dict.get("last_update_ts", now)),
                    )
                    # scarta roba scaduta
                    if (now - st.last_update_ts) <= self.ttl_seconds:
                        loaded[sym] = st
                except Exception:
                    continue

            self._states = loaded
        except Exception:
            # se file corrotto/non leggibile: non blocchiamo il backend
            self._states = {}

    # -------------------------
    # Utility
    # -------------------------
    def _cleanup_if_needed(self) -> bool:
        """
        Pulisce gli stati scaduti. Ritorna True se ha rimosso qualcosa.
        """
        now = time()
        dead = [sym for sym, st in self._states.items() if (now - st.last_update_ts) > self.ttl_seconds]
        if not dead:
            return False
        for sym in dead:
            self._states.pop(sym, None)
        return True

    def watched_symbols(self) -> Tuple[str, ...]:
        with self._lock:
            changed = self._cleanup_if_needed()
            if changed:
                self._save_to_disk()
            return tuple(self._states.keys())

    def reset(self, symbol: str) -> None:
        with self._lock:
            removed = self._states.pop(symbol, None) is not None
            if removed:
                self._save_to_disk()

    # -------------------------
    # Entry point: SOLO quando parte la simulazione
    # -------------------------
    def on_simulation_started(self, symbol: str, entry_price: float) -> str:
        """
        CHIAMALA SOLO quando:
        - mandi notifica BUY
        - e avvii davvero la simulazione

        Questo è il punto che "arruola" l'asset nel monitoraggio persistente.
        """
        if entry_price <= 0:
            return "Nota acquisto: 30% del capitale (Step 1/4)."

        with self._lock:
            self._cleanup_if_needed()

            st = ScaleState(
                step_index=1,           # Step 1 già consumato
                last_ref_price=entry_price,
                rebuy_ready=False,
                last_update_ts=time(),
            )
            self._states[symbol] = st
            self._save_to_disk()

            pct = self.tranches_pct[0]
            return f"Nota acquisto: {pct}% del capitale (Step 1/4 – notifica + simulazione avviata, monitoraggio attivo)."

    # -------------------------
    # Monitor prezzo
    # -------------------------
    def on_price_tick(self, symbol: str, price: float) -> None:
        """
        Chiamala nel tuo monitor prezzi.
        Abilita rebuy_ready se il prezzo scende rispetto a last_ref_price.
        Resetta tutto se sale abbastanza (ha preso direzione).
        """
        if price <= 0:
            return

        with self._lock:
            changed = self._cleanup_if_needed()
            st = self._states.get(symbol)
            if not st:
                if changed:
                    self._save_to_disk()
                return

            st.last_update_ts = time()

            # Drawdown trigger: abilita prossimo step
            if (not st.rebuy_ready) and price <= st.last_ref_price * (1.0 - self.dd_trigger_pct):
                st.rebuy_ready = True

            # Profit reset: se sale abbastanza, consideriamo il ciclo "risolto"
            if price >= st.last_ref_price * (1.0 + self.reset_profit_pct):
                self._states.pop(symbol, None)

            self._save_to_disk()

    # -------------------------
    # Quando trend_logic produce BUY (ma SOLO se già in monitoraggio)
    # -------------------------
    def decorate_buy_note_if_applicable(self, symbol: str, buy_price: float) -> Optional[str]:
        """
        Da chiamare quando trend_logic produce BUY.
        Ma genera una nota SOLO se:
        - symbol è già stato arruolato con on_simulation_started
        - rebuy_ready=True
        - non siamo già a 4/4
        """
        if buy_price <= 0:
            return None

        with self._lock:
            changed = self._cleanup_if_needed()
            st = self._states.get(symbol)
            if not st:
                if changed:
                    self._save_to_disk()
                return None

            st.last_update_ts = time()

            # già completato
            if st.step_index >= 4:
                self._save_to_disk()
                return None

            # avanza solo se era sceso
            if not st.rebuy_ready:
                self._save_to_disk()
                return None

            # step_index: 1-> prossimo pct = tranches[1]=20 (Step 2/4)
            pct = self.tranches_pct[st.step_index]
            step_num = st.step_index + 1

            st.step_index += 1
            st.last_ref_price = buy_price
            st.rebuy_ready = False

            self._save_to_disk()
            return f"Nota acquisto: {pct}% del capitale (Step {step_num}/4 – nuovo BUY dopo drawdown)."

    # -------------------------
    # Segnali di uscita (SELL / chiusa)
    # -------------------------
    def on_exit(self, symbol: str) -> None:
        """
        Chiamala quando chiudi la simulazione (SELL o chiusaDaBackend).
        """
        self.reset(symbol)


# Singleton
scaler = CapitalScaler()
