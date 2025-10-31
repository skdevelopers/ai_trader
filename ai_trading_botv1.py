#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, hmac, hashlib, logging, json, math, requests
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

CONFIG = {
    "mode": os.getenv("MODE", "BACKTEST"),
    "symbol": os.getenv("SYMBOL", "BTCUSDT"),
    "interval": os.getenv("INTERVAL", "5m"),
    "lookback_limit": int(os.getenv("LOOKBACK_LIMIT", "500")),
    "fees_bp": float(os.getenv("FEES_BP", "10")),
    "slippage_bp": float(os.getenv("SLIPPAGE_BP", "5")),
    "risk_per_trade": float(os.getenv("RISK_PCT", "0.0015")),
    "max_daily_loss": float(os.getenv("MAX_DAILY_LOSS", "0.01")),
    "min_atr_pct": float(os.getenv("MIN_ATR_PCT", "0.10")),
    "max_atr_pct": float(os.getenv("MAX_ATR_PCT", "1.20")),
    "ema_fast": int(os.getenv("EMA_FAST", "21")),
    "ema_slow": int(os.getenv("EMA_SLOW", "55")),
    "rsi_period": int(os.getenv("RSI_PERIOD", "14")),
    "atr_period": int(os.getenv("ATR_PERIOD", "14")),
    "atr_stop_k": float(os.getenv("ATR_STOP_K", "2.2")),
    "atr_trail_k": float(os.getenv("ATR_TRAIL_K", "1.0")),
    "tp1_R": float(os.getenv("TP1_R", "0.8")),
    "tp2_R": float(os.getenv("TP2_R", "1.6")),
    "partial_tp1": float(os.getenv("PARTIAL_TP1", "0.6")),
    "testnet_base": os.getenv("BINANCE_TESTNET", "https://testnet.binance.vision"),
    "paper_balance_usdt": float(os.getenv("PAPER_BALANCE", "1000")),
    "backtest_csv": os.getenv("BACKTEST_CSV", ""),
    "session_start_utc": int(os.getenv("SESSION_START_UTC", "12")),
    "session_end_utc": int(os.getenv("SESSION_END_UTC", "20")),
    "maker_bias_enable": os.getenv("MAKER_BIAS", "1") == "1",
    "maker_timeout_sec": int(os.getenv("MAKER_TIMEOUT_SEC", "15")),
    "maker_price_offset_bp": float(os.getenv("MAKER_OFFSET_BP", "2")),
    "ml_gate_enable": os.getenv("ML_GATE", "1") == "1",
    "ml_model_path": os.getenv("ML_MODEL_PATH", "ml_gate_model.pkl"),
    "ml_meta_path": os.getenv("ML_META_PATH", "ml_gate_meta.json"),
    "ml_ev_enable": os.getenv("ML_EV_ENABLE", "1") == "1",
    "ml_min_ev": float(os.getenv("ML_MIN_EV", "0.0")),
}

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

def ts_ms(): return int(time.time() * 1000)
def now_utc(): 
    return datetime.now(timezone.utc)
def in_session(dt_utc: datetime) -> bool:
    s, e = CONFIG["session_start_utc"], CONFIG["session_end_utc"]
    h = dt_utc.hour
    return (s <= h <= e) if s <= e else (h >= s or h <= e)

def sign_params(params, secret):
    q = "&".join([f"{k}={v}" for k,v in params.items() if v is not None])
    sig = hmac.new(secret.encode(), q.encode(), hashlib.sha256).hexdigest()
    params["signature"]=sig; return params

def ema(s: pd.Series, p:int): return s.ewm(span=p, adjust=False).mean()
def rsi(s: pd.Series, p:int=14):
    d=s.diff(); up=d.clip(lower=0.0); dn=(-d).clip(lower=0.0)
    ma_up=up.ewm(com=p-1, adjust=False).mean()
    ma_dn=dn.ewm(com=p-1, adjust=False).mean()
    rs=ma_up/(ma_dn+1e-12); return 100-(100/(1+rs))
def atr(df: pd.DataFrame, p:int=14):
    h=df["High"]; l=df["Low"]; c=df["Close"]; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def get_klines(symbol:str, interval:str, limit:int)->pd.DataFrame:
    if CONFIG["mode"]=="BACKTEST" and CONFIG["backtest_csv"]:
        df=pd.read_csv(CONFIG["backtest_csv"])
        if "OpenTime" in df.columns:
            try: df["OpenTime"]=pd.to_datetime(df["OpenTime"], unit="ms")
            except: df["OpenTime"]=pd.to_datetime(df["OpenTime"])
        return df.tail(limit)[["Open","High","Low","Close","Volume","OpenTime"]].copy()
    url=f'{CONFIG["testnet_base"]}/api/v3/klines'
    r=requests.get(url, params={"symbol":symbol,"interval":interval,"limit":min(limit,1000)}, timeout=15)
    data=r.json()
    df=pd.DataFrame(data, columns=["OpenTime","Open","High","Low","Close","Volume","CloseTime","q1","q2","q3","q4","q5"])
    df["OpenTime"]=pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    for c in ["Open","High","Low","Close","Volume"]: df[c]=df[c].astype(float)
    return df[["OpenTime","Open","High","Low","Close","Volume"]]

def detect_patterns(df: pd.DataFrame)->pd.Series:
    o,h,l,c=df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    n=len(df); sig=np.zeros(n, dtype=int)
    for i in range(1,n):
        ob,hb,lb,cb=o[i-1],h[i-1],l[i-1],c[i-1]
        oc,hc,lc,cc=o[i],h[i],l[i],c[i]
        body_prev=abs(cb-ob); body=abs(cc-oc)
        if cb<ob and cc>oc and oc<=cb and cc>=ob and body>0.5*(hb-lb): sig[i]=+1
        if cb>ob and cc<oc and oc>=cb and cc<=ob and body>0.5*(hb-lb): sig[i]=-1
        lower_wick=(oc-lc) if cc>=oc else (cc-lc)
        if (cc>oc) and lower_wick>2*body and (hc-max(cc,oc))<body: sig[i]=+1
        upper_wick=(hc-oc) if cc<=oc else (hc-cc)
        if (cc<oc) and upper_wick>2*body and (min(cc,oc)-lc)<body: sig[i]=-1
    return pd.Series(sig, index=df.index)

def apply_filters(df: pd.DataFrame, raw: pd.Series)->pd.Series:
    df=df.copy()
    df["EMA_F"]=ema(df["Close"], int(CONFIG["ema_fast"]))
    df["EMA_S"]=ema(df["Close"], int(CONFIG["ema_slow"]))
    df["RSI"]=rsi(df["Close"], int(CONFIG["rsi_period"]))
    df["ATR"]=atr(df, int(CONFIG["atr_period"]))
    df["ATR_pct"]=df["ATR"]/df["Close"]*100
    out=[]
    for i in range(len(df)):
        s=int(raw.iloc[i])
        if s==0: out.append(0); continue
        atr_ok=(df["ATR_pct"].iloc[i]>=CONFIG["min_atr_pct"]) and (df["ATR_pct"].iloc[i]<=CONFIG["max_atr_pct"])
        if not atr_ok: out.append(0); continue
        trend_up=df["EMA_F"].iloc[i]>df["EMA_S"].iloc[i]
        trend_dn=df["EMA_F"].iloc[i]<df["EMA_S"].iloc[i]
        r=df["RSI"].iloc[i]
        if s>0 and trend_up and r<65: out.append(+1)
        elif s<0 and trend_dn and r>35: out.append(-1)
        else: out.append(0)
    return pd.Series(out, index=df.index)

@dataclass
class Position:
    side: str; qty: float; entry: float; sl: float; tp1: float; tp2: float
    trail_active: bool=False; closed: bool=False; filled_tp1: bool=False

def compute_levels(price: float, atr_val: float, direction:int):
    if direction>0:
        sl=price-CONFIG["atr_stop_k"]*atr_val
        tp1=price+(price-sl)*CONFIG["tp1_R"]
        tp2=price+(price-sl)*CONFIG["tp2_R"]
    else:
        sl=price+CONFIG["atr_stop_k"]*atr_val
        tp1=price-(sl-price)*CONFIG["tp1_R"]
        tp2=price-(sl-price)*CONFIG["tp2_R"]
    return sl,tp1,tp2

def size_from_risk(balance_usdt: float, price: float, sl: float, direction:int)->float:
    risk_usd=balance_usdt*CONFIG["risk_per_trade"]
    per_unit_loss=max((price-sl) if direction>0 else (sl-price), 1e-6)
    qty=risk_usd/per_unit_loss
    return max(round(qty,6),0.0)

def account_balance_usdt()->float:
    ep=f'{CONFIG["testnet_base"]}/api/v3/account'
    params={"timestamp": int(time.time()*1000)}
    headers={"X-MBX-APIKEY": os.getenv("API_KEY","")}
    params=sign_params(params, os.getenv("API_SECRET",""))
    r=requests.get(ep, headers=headers, params=params, timeout=10)
    js=r.json()
    if "balances" not in js: return CONFIG["paper_balance_usdt"]
    for b in js["balances"]:
        if b["asset"]=="USDT": return float(b["free"])
    return CONFIG["paper_balance_usdt"]

def place_market(symbol:str, side:str, qty:float)->dict:
    ep=f'{CONFIG["testnet_base"]}/api/v3/order'
    params={"symbol":symbol,"side":side,"type":"MARKET","quantity":qty,"timestamp":int(time.time()*1000)}
    headers={"X-MBX-APIKEY": os.getenv("API_KEY","")}
    params=sign_params(params, os.getenv("API_SECRET",""))
    r=requests.post(ep, headers=headers, params=params, timeout=10)
    return r.json()

def place_limit(symbol:str, side:str, qty:float, price:float, tif="GTC")->dict:
    ep=f'{CONFIG["testnet_base"]}/api/v3/order'
    params={"symbol":symbol,"side":side,"type":"LIMIT","timeInForce":tif,"quantity":qty,"price":f"{price:.2f}","timestamp":int(time.time()*1000)}
    headers={"X-MBX-APIKEY": os.getenv("API_KEY","")}
    params=sign_params(params, os.getenv("API_SECRET",""))
    r=requests.post(ep, headers=headers, params=params, timeout=10)
    return r.json()

def open_orders(symbol:str)->list:
    ep=f'{CONFIG["testnet_base"]}/api/v3/openOrders'
    params={"symbol":symbol,"timestamp":int(time.time()*1000)}
    headers={"X-MBX-APIKEY": os.getenv("API_KEY","")}
    params=sign_params(params, os.getenv("API_SECRET",""))
    r=requests.get(ep, headers=headers, params=params, timeout=10)
    return r.json()

def cancel_order(symbol:str, orderId)->dict:
    ep=f'{CONFIG["testnet_base"]}/api/v3/order'
    params={"symbol":symbol,"orderId":orderId,"timestamp":int(time.time()*1000)}
    headers={"X-MBX-APIKEY": os.getenv("API_KEY","")}
    params=sign_params(params, os.getenv("API_SECRET",""))
    r=requests.delete(ep, headers=headers, params=params, timeout=10)
    return r.json()

def ticker_price(symbol:str)->float:
    ep=f'{CONFIG["testnet_base"]}/api/v3/ticker/price'
    r=requests.get(ep, params={"symbol":symbol}, timeout=10)
    return float(r.json()["price"])

def maker_entry(symbol:str, side:str, qty:float)->dict:
    if not CONFIG["maker_bias_enable"]:
        return place_market(symbol, side, qty)
    mid = ticker_price(symbol)
    off = CONFIG["maker_price_offset_bp"]/10000.0
    price = mid*(1 - off) if side=="BUY" else mid*(1 + off)
    lim = place_limit(symbol, side, qty, price)
    order_id = lim.get("orderId")
    start = time.time()
    while time.time() - start < CONFIG["maker_timeout_sec"]:
        time.sleep(2)
        oo = open_orders(symbol)
        still = any(o.get("orderId")==order_id for o in (oo if isinstance(oo,list) else []))
        if not still:
            return {"filled":"limit","resp":lim}
    if order_id:
        try: cancel_order(symbol, order_id)
        except: pass
    m = place_market(symbol, side, qty)
    return {"filled":"market_fallback","resp":m}

_ML = {"loaded": False}
def load_gate():
    if _ML["loaded"]: return
    try:
        import pickle
        with open(CONFIG["ml_model_path"], "rb") as f:
            PKL = pickle.load(f)
        _ML["clf"] = PKL["clf"]
        _ML["features"] = PKL["features"]
        _ML["threshold"] = PKL.get("threshold", 0.6)
        meta = {}
        try:
            with open(CONFIG["ml_meta_path"], "r") as g:
                meta = json.load(g)
        except Exception:
            meta = {}
        _ML["avg_win_R"] = float(meta.get("avg_win_R", 1.0))
        _ML["avg_loss_R"] = abs(float(meta.get("avg_loss_R", 1.0)))
    except Exception as e:
        _ML["clf"]=None; _ML["features"]=[]; _ML["threshold"]=1.0; _ML["avg_win_R"]=1.0; _ML["avg_loss_R"]=1.0
    _ML["loaded"]=True

def featurize_row(row: dict)->dict:
    rng=max(row["high"]-row["low"], 1e-8)
    body=abs(row["close"]-row["open"])
    return {
        "body_pct": body/rng,
        "upper_wick_pct": (row["high"]-max(row["close"], row["open"])) / rng,
        "lower_wick_pct": (min(row["close"], row["open"])-row["low"]) / rng,
        "dist_fast": (row["close"]-row["ema_f"])/row["close"],
        "dist_slow": (row["close"]-row["ema_s"])/row["close"],
        "ema_fast_slope": row.get("ema_f_slope",0.0),
        "ema_slow_slope": row.get("ema_s_slope",0.0),
        "bb_width": row.get("bb_width",0.0),
        "atr_pct": row["atr_pct"],
        "rsi": row["rsi"],
        "sig_dir": row["signal"],
        "f_bull_eng": row.get("f_bull_eng",0), "f_bear_eng": row.get("f_bear_eng",0),
        "f_hammer": row.get("f_hammer",0), "f_shooting": row.get("f_shooting",0),
        "f_inside": row.get("f_inside",0), "f_nr4": row.get("f_nr4",0), "f_nr7": row.get("f_nr7",0),
        "f_sfp_bull": row.get("f_sfp_bull",0), "f_sfp_bear": row.get("f_sfp_bear",0),
    }

def gate_ok(row: dict)->bool:
    if not CONFIG["ml_gate_enable"]:
        return True
    load_gate()
    clf=_ML["clf"]
    if clf is None: return True
    import numpy as np
    X=np.array([[row.get(k,0.0) for k in _ML["features"]]], dtype=float)
    p=float(clf.predict_proba(X)[:,1][0])
    if p<=_ML["threshold"]: return False
    if CONFIG["ml_ev_enable"]:
        ev = p*_ML["avg_win_R"] - (1-p)*_ML["avg_loss_R"]
        return ev >= CONFIG["ml_min_ev"]
    return True

class Backtester:
    def __init__(self, df: pd.DataFrame):
        self.df=df.copy().reset_index(drop=True)
        self.balance=CONFIG["paper_balance_usdt"]
        self.daily_start=self.balance
        self.pos: Optional[Position]=None
        self.eq=[]

    def run(self, f: pd.DataFrame, signals: pd.Series):
        self.df=f.reset_index(drop=True)
        self.eq=[]; self.pos=None
        for i in range(len(self.df)):
            self.step(i, int(signals.iloc[i]))
        equity=pd.DataFrame({"time": self.df["OpenTime"], "equity": self.eq})
        peak=equity["equity"].cummax()
        dd=(equity["equity"]-peak)/peak.replace(0,np.nan)
        result={
            "final_balance": float(self.eq[-1]),
            "total_return": float(self.eq[-1]/self.eq[0]-1.0) if self.eq else 0.0,
            "max_drawdown": float(dd.min()) if len(dd) else 0.0,
            "equity": equity,
        }
        return result

    def step(self, i:int, signal:int):
        row=self.df.iloc[i]; price=row["Close"]; a=row["ATR"]
        if not in_session(row["OpenTime"].to_pydatetime().replace(tzinfo=timezone.utc)):
            self.eq.append(self.balance); return
        if i>0 and self.df["OpenTime"].iloc[i].date()!=self.df["OpenTime"].iloc[i-1].date():
            self.daily_start=self.balance
        if (self.daily_start-self.balance)/max(self.daily_start,1e-9)>=CONFIG["max_daily_loss"]:
            self.eq.append(self.balance); return
        fee=CONFIG["fees_bp"]/10000.0; slip=CONFIG["slippage_bp"]/10000.0

        if self.pos and not self.pos.closed:
            if self.pos.side=="LONG":
                if row["Low"]<=self.pos.sl:
                    exit_price=self.pos.sl*(1-slip)
                    pnl=(exit_price-self.pos.entry)*self.pos.qty - fee*exit_price*self.pos.qty
                    self.balance+=pnl; self.pos.closed=True
                elif (not self.pos.filled_tp1) and row["High"]>=self.pos.tp1:
                    dq=self.pos.qty*CONFIG["partial_tp1"]
                    exit_price=self.pos.tp1*(1-slip)
                    pnl=(exit_price-self.pos.entry)*dq - fee*exit_price*dq
                    self.balance+=pnl; self.pos.qty-=dq; self.pos.filled_tp1=True; self.pos.trail_active=True
                elif row["High"]>=self.pos.tp2:
                    exit_price=self.pos.tp2*(1-slip)
                    pnl=(exit_price-self.pos.entry)*self.pos.qty - fee*exit_price*self.pos.qty
                    self.balance+=pnl; self.pos.closed=True
                if self.pos.trail_active and not self.pos.closed:
                    self.pos.sl=max(self.pos.sl, price-CONFIG["atr_trail_k"]*a)

        if (self.pos is None or self.pos.closed) and signal!=0:
            direction=1 if signal>0 else -1
            feat = featurize_row({
                "open": float(row["Open"]), "high": float(row["High"]), "low": float(row["Low"]), "close": float(row["Close"]),
                "ema_f": float(self.df["EMA_F"].iloc[i]), "ema_s": float(self.df["EMA_S"].iloc[i]),
                "ema_f_slope": float(self.df["EMA_F"].iloc[i]-self.df["EMA_F"].iloc[i-1]) if i>0 else 0.0,
                "ema_s_slope": float(self.df["EMA_S"].iloc[i]-self.df["EMA_S"].iloc[i-1]) if i>0 else 0.0,
                "bb_width": float(self.df["Close"].rolling(20).std().iloc[i]*4 / (self.df["Close"].rolling(20).mean().iloc[i]+1e-12) if i>=20 else 0.0),
                "atr_pct": float(self.df["ATR_pct"].iloc[i]), "rsi": float(self.df["RSI"].iloc[i]), "signal": int(signal),
            })
            if not gate_ok(feat):
                self.eq.append(self.balance); return
            sl,tp1,tp2=compute_levels(price, a, direction)
            if sl>=price: self.eq.append(self.balance); return
            qty=size_from_risk(self.balance, price, sl, direction); qty=max(round(qty,6),0.0)
            if qty<=0: self.eq.append(self.balance); return
            fill=price*(1+(CONFIG["slippage_bp"]/10000.0))
            trade_fee=CONFIG["fees_bp"]/10000.0 * fill * qty
            self.balance-=trade_fee
            self.pos=Position("LONG", qty, fill, sl, tp1, tp2)
        self.eq.append(self.balance)

def build_signal_frame(df: pd.DataFrame):
    raw=detect_patterns(df)
    f=df.copy()
    f["EMA_F"]=ema(f["Close"], int(CONFIG["ema_fast"]))
    f["EMA_S"]=ema(f["Close"], int(CONFIG["ema_slow"]))
    f["RSI"]=rsi(f["Close"], int(CONFIG["rsi_period"]))
    f["ATR"]=atr(f, int(CONFIG["atr_period"]))
    f["ATR_pct"]=f["ATR"]/f["Close"]*100
    sig=apply_filters(f, raw)
    return f, sig

def backtest_main():
    df=get_klines(CONFIG["symbol"], CONFIG["interval"], CONFIG["lookback_limit"])
    f,sig=build_signal_frame(df)
    bt=Backtester(f)
    res=bt.run(f, sig)
    print(f"Backtest Final Balance: {res['final_balance']:.2f} | Return: {res['total_return']*100:.2f}% | MaxDD: {res['max_drawdown']*100:.2f}%")
    res["equity"].to_csv("equity_curve.csv", index=False)
    with open("backtest_summary.json","w") as fp:
        json.dump({k:(v if not isinstance(v, pd.DataFrame) else None) for k,v in res.items()}, fp, indent=2)

if __name__=="__main__":
    print("CONFIG:", json.dumps(CONFIG, indent=2))
    if CONFIG["mode"].upper()=="LIVE":
        pass  # live loop omitted in this compact file; use the larger version if needed
    else:
        backtest_main()
