#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import hmac
import hashlib
import logging
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable

import numpy as np
import pandas as pd
import requests
import certifi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# progress + plots (optional)
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ========= OPTIONAL ML GATE =========
USE_ML_GATE_DEFAULT = "true"
_ML_GATE_AVAILABLE = False
try:
    from ml_gate import featurize_row, pass_gate
    _ML_GATE_AVAILABLE = True
except Exception:
    _ML_GATE_AVAILABLE = False

pd.options.mode.copy_on_write = True

def _get_env_float(key: str, default: str) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return float(default)

def _get_env_int(key: str, default: str) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return int(default)

CONFIG = {
    "mode": os.getenv("MODE", "BACKTEST").upper(),
    "symbol": os.getenv("SYMBOL", "BTCUSDT"),
    "interval": os.getenv("INTERVAL", "5m"),
    "lookback_limit": _get_env_int("LOOKBACK_LIMIT", "1500"),

    "fees_bp": _get_env_float("FEES_BP", "10"),
    "slippage_bp": _get_env_float("SLIPPAGE_BP", "5"),

    "risk_per_trade": _get_env_float("RISK_PCT", "0.003"),
    "max_daily_loss": _get_env_float("MAX_DAILY_LOSS", "0.02"),
    "max_positions": _get_env_int("MAX_POSITIONS", "1"),

    "min_atr_pct": _get_env_float("MIN_ATR_PCT", "0.15"),
    "max_atr_pct": _get_env_float("MAX_ATR_PCT", "2.0"),
    "ema_fast": _get_env_int("EMA_FAST", "20"),
    "ema_slow": _get_env_int("EMA_SLOW", "50"),
    "rsi_period": _get_env_int("RSI_PERIOD", "14"),
    "atr_period": _get_env_int("ATR_PERIOD", "14"),
    "atr_stop_k": _get_env_float("ATR_STOP_K", "2.0"),
    "atr_trail_k": _get_env_float("ATR_TRAIL_K", "1.0"),
    "tp1_R": _get_env_float("TP1_R", "1.0"),
    "tp2_R": _get_env_float("TP2_R", "2.0"),
    "partial_tp1": _get_env_float("PARTIAL_TP1", "0.5"),

    "testnet_base": os.getenv("BASE_URL", os.getenv("BINANCE_TESTNET", "https://testnet.binance.vision")),
    "verify_ssl": os.getenv("VERIFY_SSL", "true").lower() != "false",
    "paper_balance_usdt": _get_env_float("PAPER_BALANCE", "1000"),
    "backtest_csv": os.getenv("BACKTEST_CSV", ""),
    "cache_dir": os.getenv("CACHE_DIR", "data/cache"),
    "user_agent": os.getenv("USER_AGENT", "SKDev-AITrader/1.2 (+Win10)"),

    "use_ml_gate": os.getenv("USE_ML_GATE", USE_ML_GATE_DEFAULT).lower() == "true",
    "ml_gate_in_backtest": os.getenv("ML_GATE_IN_BACKTEST", "false").lower() == "true",

    "show_progress": os.getenv("SHOW_PROGRESS", "true").lower() == "true",
    "heartbeat_every": _get_env_int("HEARTBEAT_EVERY", "250"),
    "equity_png": os.getenv("EQUITY_PNG", "equity_curve.png"),
}

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

os.makedirs("logs", exist_ok=True)
os.makedirs(CONFIG["cache_dir"], exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": CONFIG["user_agent"], "Accept": "application/json"})
    return s

SESSION = build_session()
SSL_VERIFY = certifi.where() if CONFIG["verify_ssl"] else False

def ts_ms() -> int:
    return int(time.time() * 1000)

def sign_params(params: Dict[str, str], secret: str) -> Dict[str, str]:
    q = "&".join([f"{k}={v}" for k, v in params.items() if v is not None])
    sig = hmac.new(secret.encode(), q.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params

REQ_COLS = ["OpenTime", "Open", "High", "Low", "Close", "Volume"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c.lower(): c for c in df.columns}
    rename = {}
    for want in ["opentime","open","high","low","close","volume","time","date"]:
        if want in mapping:
            rename[mapping[want]] = {
                "opentime":"OpenTime","open":"Open","high":"High","low":"Low","close":"Close",
                "volume":"Volume","time":"OpenTime","date":"OpenTime"
            }[want]
    return df.rename(columns=rename)

def normalize_ohlcv(df_in: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df_in.copy())
    if "OpenTime" not in df.columns:
        raise ValueError("Input data missing 'OpenTime' (or alias 'time'/'date').")
    if not np.issubdtype(df["OpenTime"].dtype, np.datetime64):
        try:
            df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", errors="raise")
        except Exception:
            df["OpenTime"] = pd.to_datetime(df["OpenTime"], errors="coerce")
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            raise ValueError(f"Input data missing '{col}' column.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["OpenTime","Open","High","Low","Close","Volume"])
    df = df.sort_values("OpenTime").reset_index(drop=True)
    return df[REQ_COLS]

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff().to_numpy()
    up = np.clip(d, 0, None)
    dn = np.clip(-d, 0, None)
    up_ema = pd.Series(up).ewm(com=p-1, adjust=False).mean()
    dn_ema = pd.Series(dn).ewm(com=p-1, adjust=False).mean()
    rs = up_ema / (dn_ema + 1e-12)
    rsi_vals = 100 - (100 / (1 + rs))
    return pd.Series(rsi_vals.values, index=s.index)

def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([np.abs(h - l), np.abs(h - pc), np.abs(l - pc)])
    atr_series = pd.Series(tr).ewm(alpha=1/p, adjust=False).mean()
    return pd.Series(atr_series.values, index=df.index)

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "1d": 86_400_000
}
def _klines_url() -> str: return f'{CONFIG["testnet_base"].rstrip("/")}/api/v3/klines'
def _read_csv_any(path: str) -> pd.DataFrame: return normalize_ohlcv(pd.read_csv(path))

def _read_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        if "OpenTime" in df.columns and not np.issubdtype(df["OpenTime"].dtype, np.datetime64):
            try: df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
            except Exception: df["OpenTime"] = pd.to_datetime(df["OpenTime"])
        return normalize_ohlcv(df)
    except Exception:
        return None

def _write_cache(df: pd.DataFrame, path: str) -> None:
    try:
        tmp = df.copy()
        if pd.api.types.is_datetime64_any_dtype(tmp["OpenTime"]):
            tmp["OpenTime"] = (tmp["OpenTime"].astype("int64") // 10**6)
        tmp.to_csv(path, index=False)
    except Exception as e:
        logging.warning(f"Cache write failed: {e}")

def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    if CONFIG["mode"] == "BACKTEST" and CONFIG["backtest_csv"]:
        return _read_csv_any(CONFIG["backtest_csv"]).tail(limit).copy()

    cache_name = f'{symbol}_{interval}_{limit}.csv'
    cache_path = os.path.join(CONFIG["cache_dir"], cache_name)
    cached = _read_cache(cache_path)
    if cached is not None and len(cached) >= min(limit, 1000):
        return cached.tail(limit).copy()

    url = _klines_url(); per_req = 1000; remaining = limit
    all_rows = []; end_ms = None; tries = 0

    while remaining > 0:
        fetch = min(per_req, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": fetch}
        if end_ms is not None: params["endTime"] = end_ms
        try:
            r = SESSION.get(url, params=params, timeout=15, verify=SSL_VERIFY)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0: break
            part = pd.DataFrame(data, columns=["OpenTime","Open","High","Low","Close","Volume","CloseTime","q1","q2","q3","q4","q5"])
            part["OpenTime"] = pd.to_datetime(part["OpenTime"], unit="ms")
            for c in ["Open","High","Low","Close","Volume"]: part[c] = part[c].astype(float)
            part = part[["OpenTime","Open","High","Low","Close","Volume"]]
            all_rows.append(part)
            remaining -= len(part)
            end_ms = int((part["OpenTime"].iloc[0].value // 10**6) - INTERVAL_MS.get(interval, 60_000))
            time.sleep(0.2)
        except requests.exceptions.SSLError as e:
            logging.error(f"SSL error: {e}. Set VERIFY_SSL=false only for testing."); raise
        except requests.exceptions.RequestException as e:
            tries += 1; logging.warning(f"Network error ({tries}/3): {e}")
            if tries >= 3: raise
            time.sleep(1.0)
        if end_ms is not None and len(all_rows) > 0 and len(all_rows[-1]) < fetch: break

    if not all_rows: raise RuntimeError("No kline data retrieved.")
    df = pd.concat(all_rows, axis=0).sort_values("OpenTime").reset_index(drop=True)
    df = normalize_ohlcv(df); _write_cache(df, cache_path)
    return df.tail(limit).copy()

def detect_patterns(df: pd.DataFrame) -> pd.Series:
    o = df["Open"].to_numpy(); h = df["High"].to_numpy(); l = df["Low"].to_numpy(); c = df["Close"].to_numpy()
    n = len(df); sig = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        ob, hb, lb, cb = o[i-1], h[i-1], l[i-1], c[i-1]
        oc, hc, lc, cc = o[i], h[i], l[i], c[i]
        body = abs(cc - oc)
        if cb < ob and cc > oc and oc <= cb and cc >= ob and body > 0.5 * (hb - lb): sig[i] = 1
        if cb > ob and cc < oc and oc >= cb and cc <= ob and body > 0.5 * (hb - lb): sig[i] = -1
        lower_wick = (oc - lc) if cc >= oc else (cc - lc)
        if (cc > oc) and lower_wick > 2 * body and (hc - max(cc, oc)) < body: sig[i] = 1
        upper_wick = (hc - oc) if cc <= oc else (hc - cc)
        if (cc < oc) and upper_wick > 2 * body and (min(cc, oc) - lc) < body: sig[i] = -1
    return pd.Series(sig, index=df.index, dtype="int8")

def apply_filters(df: pd.DataFrame, raw: pd.Series) -> pd.Series:
    f = df.copy()
    f["EMA_F"] = ema(f["Close"], int(CONFIG["ema_fast"]))
    f["EMA_S"] = ema(f["Close"], int(CONFIG["ema_slow"]))
    f["RSI"]   = rsi(f["Close"], int(CONFIG["rsi_period"]))
    f["ATR"]   = atr(f, int(CONFIG["atr_period"]))
    f["ATR_pct"] = f["ATR"] / f["Close"] * 100

    out = np.zeros(len(f), dtype="int8")
    ema_f = f["EMA_F"].to_numpy(); ema_s = f["EMA_S"].to_numpy()
    rsi_v = f["RSI"].to_numpy(); atrp  = f["ATR_pct"].to_numpy(); raw_v = raw.to_numpy()
    for i in range(len(f)):
        s = int(raw_v[i]); if_zero = (s == 0)
        if if_zero: continue
        if not (CONFIG["min_atr_pct"] <= atrp[i] <= CONFIG["max_atr_pct"]): continue
        trend_up = ema_f[i] > ema_s[i]; trend_dn = ema_f[i] < ema_s[i]; r = rsi_v[i]
        if s > 0 and trend_up and r < 60: out[i] = 1
        elif s < 0 and trend_dn and r > 40: out[i] = -1
    return pd.Series(out, index=f.index, dtype="int8")

@dataclass
class Position:
    side: str
    qty: float
    entry: float
    sl: float
    tp1: float
    tp2: float
    trail_active: bool = False
    closed: bool = False
    realized_pnl: float = 0.0
    filled_tp1: bool = False

def compute_levels(price: float, atr_val: float, direction: int) -> Tuple[float, float, float]:
    if direction > 0:
        sl = price - CONFIG["atr_stop_k"] * atr_val
        tp1 = price + (price - sl) * CONFIG["tp1_R"]
        tp2 = price + (price - sl) * CONFIG["tp2_R"]
    else:
        sl = price + CONFIG["atr_stop_k"] * atr_val
        tp1 = price - (sl - price) * CONFIG["tp1_R"]
        tp2 = price - (sl - price) * CONFIG["tp2_R"]
    return sl, tp1, tp2

def size_from_risk(balance_usdt: float, price: float, sl: float, direction: int) -> float:
    risk_usd = balance_usdt * CONFIG["risk_per_trade"]
    per_unit_loss = max((price - sl) if direction > 0 else (sl - price), 1e-6)
    qty = risk_usd / per_unit_loss
    return max(round(qty, 6), 0.0)

def account_balance_usdt() -> float:
    ep = f'{CONFIG["testnet_base"].rstrip("/")}/api/v3/account'
    params = {"timestamp": ts_ms()}; headers = {"X-MBX-APIKEY": API_KEY}
    params = sign_params(params, API_SECRET)
    r = SESSION.get(ep, headers=headers, params=params, timeout=10, verify=SSL_VERIFY)
    js = r.json()
    if "balances" not in js: raise RuntimeError(js)
    for b in js["balances"]:
        if b.get("asset") == "USDT": return float(b.get("free", 0.0))
    return 0.0

def place_market(symbol: str, side: str, qty: float) -> dict:
    ep = f'{CONFIG["testnet_base"].rstrip("/")}/api/v3/order'
    params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty, "timestamp": ts_ms()}
    headers = {"X-MBX-APIKEY": API_KEY}
    params = sign_params(params, API_SECRET)
    r = SESSION.post(ep, headers=headers, params=params, timeout=10, verify=SSL_VERIFY)
    return r.json()

def build_feature_row(f: pd.DataFrame, last_sig: int) -> Dict[str, float]:
    if len(f) < 21:
        last = f.iloc[-1]
        return {
            "open": float(last["Open"]), "high": float(last["High"]), "low": float(last["Low"]), "close": float(last["Close"]),
            "ema_fast": float(last["EMA_F"]), "ema_slow": float(last["EMA_S"]),
            "ema_fast_slope": 0.0, "ema_slow_slope": 0.0, "bb_width": 0.0,
            "atr_pct": float(last["ATR"] / (last["Close"] + 1e-12) * 100.0),
            "rsi": float(last["RSI"]), "signal": int(last_sig),
            "f_bull_eng": 0, "f_bear_eng": 0, "f_hammer": 0, "f_shooting": 0,
            "f_inside": 0, "f_nr4": 0, "f_nr7": 0, "f_sfp_bull": 0, "f_sfp_bear": 0,
        }
    last = f.iloc[-1]
    ema_fast_slope = float(f["EMA_F"].iloc[-1] - f["EMA_F"].iloc[-2])
    ema_slow_slope = float(f["EMA_S"].iloc[-1] - f["EMA_S"].iloc[-2])
    bb_width = float((f["Close"].rolling(20).std()*4 / (f["Close"].rolling(20).mean() + 1e-12)).iloc[-1])
    return {
        "open": float(last["Open"]), "high": float(last["High"]), "low": float(last["Low"]), "close": float(last["Close"]),
        "ema_fast": float(last["EMA_F"]), "ema_slow": float(last["EMA_S"]),
        "ema_fast_slope": ema_fast_slope, "ema_slow_slope": ema_slow_slope, "bb_width": bb_width,
        "atr_pct": float(last["ATR"] / (last["Close"] + 1e-12) * 100.0),
        "rsi": float(last["RSI"]), "signal": int(last_sig),
        "f_bull_eng": 0, "f_bear_eng": 0, "f_hammer": 0, "f_shooting": 0,
        "f_inside": 0, "f_nr4": 0, "f_nr7": 0, "f_sfp_bull": 0, "f_sfp_bear": 0,
    }

def ml_gate_allows_entry(f: pd.DataFrame, last_sig: int) -> bool:
    use_in_this_mode = CONFIG["use_ml_gate"] and (_ML_GATE_AVAILABLE) and (
        CONFIG["mode"] == "LIVE" or CONFIG["ml_gate_in_backtest"]
    )
    if not use_in_this_mode:
        return True
    try:
        row = build_feature_row(f, last_sig)
        feat = featurize_row(row)
        allowed = pass_gate(feat)
        if not allowed:
            logging.info("ML gate filtered out this entry.")
        return bool(allowed)
    except Exception as e:
        logging.warning(f"ML gate error -> bypassing gate: {e}")
        return True

class Backtester:
    def __init__(self, df: pd.DataFrame, show_progress: bool = True):
        self.df = df.copy().reset_index(drop=True)
        self.balance = CONFIG["paper_balance_usdt"]
        self.daily_start = self.balance
        self.pos: Optional[Position] = None
        self.eq = []
        self.show_progress = show_progress and _HAS_TQDM

    def step(self, i: int, signal: int):
        row = self.df.iloc[i]
        price = row["Close"]; a = row["ATR"]

        if i > 0 and self.df["OpenTime"].iloc[i].date() != self.df["OpenTime"].iloc[i-1].date():
            self.daily_start = self.balance
        if (self.daily_start - self.balance) / max(self.daily_start, 1e-9) >= CONFIG["max_daily_loss"]:
            self.eq.append(self.balance); return

        fee = CONFIG["fees_bp"] / 10000.0; slip = CONFIG["slippage_bp"] / 10000.0

        if self.pos and not self.pos.closed:
            if self.pos.side == "LONG":
                if row["Low"] <= self.pos.sl:
                    exit_price = self.pos.sl * (1 - slip)
                    pnl = (exit_price - self.pos.entry) * self.pos.qty - fee * exit_price * self.pos.qty
                    self.balance += pnl; self.pos.closed = True
                elif (not self.pos.filled_tp1) and row["High"] >= self.pos.tp1:
                    dq = self.pos.qty * CONFIG["partial_tp1"]
                    exit_price = self.pos.tp1 * (1 - slip)
                    pnl = (exit_price - self.pos.entry) * dq - fee * exit_price * dq
                    self.balance += pnl; self.pos.qty -= dq; self.pos.filled_tp1 = True; self.pos.trail_active = True
                elif row["High"] >= self.pos.tp2:
                    exit_price = self.pos.tp2 * (1 - slip)
                    pnl = (exit_price - self.pos.entry) * self.pos.qty - fee * exit_price * self.pos.qty
                    self.balance += pnl; self.pos.closed = True
                if self.pos.trail_active and not self.pos.closed:
                    self.pos.sl = max(self.pos.sl, price - CONFIG["atr_trail_k"] * a)
            else:
                if row["High"] >= self.pos.sl:
                    exit_price = self.pos.sl * (1 + slip)
                    pnl = (self.pos.entry - exit_price) * self.pos.qty - fee * exit_price * self.pos.qty
                    self.balance += pnl; self.pos.closed = True
                elif (not self.pos.filled_tp1) and row["Low"] <= self.pos.tp1:
                    dq = self.pos.qty * CONFIG["partial_tp1"]
                    exit_price = self.pos.tp1 * (1 + slip)
                    pnl = (self.pos.entry - exit_price) * dq - fee * exit_price * dq
                    self.balance += pnl; self.pos.qty -= dq; self.pos.filled_tp1 = True; self.pos.trail_active = True
                elif row["Low"] <= self.pos.tp2:
                    exit_price = self.pos.tp2 * (1 + slip)
                    pnl = (self.pos.entry - exit_price) * self.pos.qty - fee * exit_price * self.pos.qty
                    self.balance += pnl; self.pos.closed = True
                if self.pos.trail_active and not self.pos.closed:
                    self.pos.sl = min(self.pos.sl, price + CONFIG["atr_trail_k"] * a)

        if (self.pos is None or self.pos.closed) and signal != 0:
            direction = 1 if signal > 0 else -1
            sl, tp1, tp2 = compute_levels(price, a, direction)
            if (direction > 0 and sl >= price) or (direction < 0 and sl <= price):
                self.eq.append(self.balance); return
            qty = size_from_risk(self.balance, price, sl, direction)
            if qty <= 0:
                self.eq.append(self.balance); return
            fill = price * (1 + (CONFIG["slippage_bp"] / 10000.0) * (1 if direction > 0 else -1))
            trade_fee = CONFIG["fees_bp"] / 10000.0 * fill * qty
            self.balance -= trade_fee
            self.pos = Position("LONG" if direction > 0 else "SHORT", qty, fill, sl, tp1, tp2)

        self.eq.append(self.balance)

    def run(self, df: pd.DataFrame, sig: pd.Series, on_bar: Optional[Callable[[int, int], None]] = None) -> Dict:
        total = len(df)
        iterator = range(total)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Backtest", unit="bar")
        hb_every = max(1, int(CONFIG["heartbeat_every"]))
        for i in iterator:
            if (not self.show_progress) and (i % hb_every == 0):
                print(f"Backtest progress: {i}/{total} bars", flush=True)
            self.step(i, int(sig.iloc[i]))
            if on_bar:
                on_bar(i + 1, total)
        res = pd.DataFrame({"time": self.df["OpenTime"], "equity": self.eq})
        total_ret = (self.balance - CONFIG["paper_balance_usdt"]) / CONFIG["paper_balance_usdt"]
        dd = (res["equity"].cummax() - res["equity"]) / res["equity"].cummax()
        return {"final_balance": self.balance, "total_return": total_ret, "max_drawdown": float(dd.max() if len(dd) else 0.0), "equity": res}

def build_signal_frame(df: pd.DataFrame):
    f = normalize_ohlcv(df)
    f["EMA_F"] = ema(f["Close"], int(CONFIG["ema_fast"]))
    f["EMA_S"] = ema(f["Close"], int(CONFIG["ema_slow"]))
    f["RSI"]   = rsi(f["Close"], int(CONFIG["rsi_period"]))
    f["ATR"]   = atr(f, int(CONFIG["atr_period"]))
    f["ATR_pct"] = f["ATR"] / f["Close"] * 100
    raw = detect_patterns(f)
    sig = apply_filters(f, raw)
    return f, sig

def save_equity_png(df_equity: pd.DataFrame, path: str) -> None:
    if not _HAS_MPL:
        logging.info("matplotlib not available; skipping PNG chart."); return
    try:
        plt.figure(); plt.plot(df_equity["time"], df_equity["equity"])
        plt.title("Equity Curve"); plt.xlabel("Time"); plt.ylabel("Equity (USDT)")
        plt.tight_layout(); plt.savefig(path); plt.close()
        logging.info(f"Saved equity chart to {path}")
    except Exception as e:
        logging.warning(f"Failed to save equity chart: {e}")

def backtest_main(show_progress: bool, save_plot: bool):
    df = get_klines(CONFIG["symbol"], CONFIG["interval"], CONFIG["lookback_limit"])
    f, sig = build_signal_frame(df)
    bt = Backtester(f, show_progress=show_progress)
    res = bt.run(f, sig)
    print(f"Backtest Final Balance: {res['final_balance']:.2f} | Return: {res['total_return']*100:.2f}% | MaxDD: {res['max_drawdown']*100:.2f}%")
    res["equity"].to_csv("equity_curve.csv", index=False)
    with open("backtest_summary.json", "w") as fp:
        json.dump({k:(v if not isinstance(v, pd.DataFrame) else None) for k,v in res.items()}, fp, indent=2)
    if save_plot: save_equity_png(res["equity"], CONFIG["equity_png"])
    print("DONE.")

def live_loop():
    if not (API_KEY and API_SECRET):
        raise SystemExit("LIVE mode requires API_KEY and API_SECRET in env.")
    daily_start = None; open_pos: Optional[Position] = None

    while True:
        try:
            df = get_klines(CONFIG["symbol"], CONFIG["interval"], CONFIG["lookback_limit"])
            f, sig = build_signal_frame(df)
            last = f.iloc[-1]; last_sig = int(sig.iloc[-1])
            price = float(last["Close"]); a = float(last["ATR"])

            now_date = f["OpenTime"].iloc[-1].date()
            prev_date = f["OpenTime"].iloc[-2].date() if len(f) >= 2 else now_date
            if daily_start is None or now_date != prev_date:
                daily_start = account_balance_usdt()
            bal_now = account_balance_usdt()
            if bal_now < daily_start * (1 - CONFIG["max_daily_loss"]):
                logging.info("Max daily loss reached; pausing 60s."); time.sleep(60); continue

            if open_pos:
                if open_pos.trail_active:
                    open_pos.sl = max(open_pos.sl, price - CONFIG["atr_trail_k"] * a)
                if price <= open_pos.sl:
                    qty = round(open_pos.qty, 6); r = place_market(CONFIG["symbol"], "SELL", qty)
                    logging.info(f"STOP SELL {r}"); open_pos = None
                elif (not open_pos.filled_tp1) and price >= open_pos.tp1:
                    qty = round(open_pos.qty * CONFIG["partial_tp1"], 6)
                    r = place_market(CONFIG["symbol"], "SELL", qty)
                    logging.info(f"TP1 SELL {r}")
                    open_pos.qty -= qty; open_pos.filled_tp1 = True; open_pos.trail_active = True
                elif price >= open_pos.tp2:
                    qty = round(open_pos.qty, 6); r = place_market(CONFIG["symbol"], "SELL", qty)
                    logging.info(f"TP2 SELL {r}"); open_pos = None

            if (open_pos is None) and last_sig > 0:
                if not ml_gate_allows_entry(f, last_sig):
                    time.sleep(10); continue
                sl, tp1, tp2 = compute_levels(price, a, +1)
                bal = account_balance_usdt()
                qty = size_from_risk(bal, price, sl, +1); qty = max(round(qty, 6), 0.0)
                if qty > 0:
                    r = place_market(CONFIG["symbol"], "BUY", qty)
                    logging.info(f"ENTRY BUY {r}")
                    open_pos = Position("LONG", qty, price, sl, tp1, tp2)

            time.sleep(10)
        except KeyboardInterrupt:
            logging.info("Interrupted; exiting live loop."); break
        except Exception as e:
            logging.exception(e); time.sleep(5)

def parse_args():
    p = argparse.ArgumentParser(description="AI Trading Bot")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars (backtest only)")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    p.add_argument("--plot", action="store_true", help="Save equity curve PNG (backtest only)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    show_progress = CONFIG["show_progress"]
    if args.progress: show_progress = True
    if args.no_progress: show_progress = False
    save_plot = args.plot

    safe_cfg = {**CONFIG, "API_KEY":"***" if API_KEY else "", "API_SECRET":"***" if API_SECRET else ""}
    print("CONFIG:", json.dumps(safe_cfg, indent=2))
    if CONFIG["mode"] == "LIVE":
        if CONFIG["use_ml_gate"] and not _ML_GATE_AVAILABLE:
            logging.warning("USE_ML_GATE=true but ml_gate module not found; continuing without ML gate.")
        live_loop()
    else:
        backtest_main(show_progress=show_progress, save_plot=save_plot)
