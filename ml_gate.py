# ml_gate.py
import pickle, numpy as np

with open("ml_gate_model.pkl", "rb") as f:
    PKL = pickle.load(f)

CLF = PKL["clf"]
FEATURES = PKL["features"]
THRESH = PKL["threshold"]

def featurize_row(row) -> dict:
    # expects a dict-like "row" with the following keys already computed:
    # open, high, low, close, ema_fast, ema_slow, ema_fast_slope, ema_slow_slope,
    # bb_width, atr_pct, rsi, signal and pattern flags below:
    # f_bull_eng, f_bear_eng, f_hammer, f_shooting, f_inside, f_nr4, f_nr7, f_sfp_bull, f_sfp_bear
    rng = max(row["high"] - row["low"], 1e-8)
    body = abs(row["close"] - row["open"])
    body_pct = body / rng
    upper_wick_pct = (row["high"] - max(row["close"], row["open"])) / rng
    lower_wick_pct = (min(row["close"], row["open"]) - row["low"]) / rng
    dist_fast = (row["close"] - row["ema_fast"]) / row["close"]
    dist_slow = (row["close"] - row["ema_slow"]) / row["close"]

    feat = {
        "body_pct": body_pct,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "dist_fast": dist_fast,
        "dist_slow": dist_slow,
        "ema_fast_slope": row["ema_fast_slope"],
        "ema_slow_slope": row["ema_slow_slope"],
        "bb_width": row["bb_width"],
        "atr_pct": row["atr_pct"],
        "rsi": row["rsi"],
        "sig_dir": row["signal"],
        "f_bull_eng": row.get("f_bull_eng", 0),
        "f_bear_eng": row.get("f_bear_eng", 0),
        "f_hammer": row.get("f_hammer", 0),
        "f_shooting": row.get("f_shooting", 0),
        "f_inside": row.get("f_inside", 0),
        "f_nr4": row.get("f_nr4", 0),
        "f_nr7": row.get("f_nr7", 0),
        "f_sfp_bull": row.get("f_sfp_bull", 0),
        "f_sfp_bear": row.get("f_sfp_bear", 0),
    }
    return feat

def pass_gate(row: dict) -> bool:
    import numpy as np
    x = np.array([[row.get(k, 0.0) for k in FEATURES]], dtype=float)
    p = float(CLF.predict_proba(x)[:,1][0])
    return p > THRESH
