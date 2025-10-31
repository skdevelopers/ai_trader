# backtest_compare.py
import json, numpy as np, pandas as pd
import ai_trading_bot as bot
from ml_gate import featurize_row, pass_gate

def run_once(use_gate: bool):
    df = bot.get_klines(bot.CONFIG["symbol"], bot.CONFIG["interval"], bot.CONFIG["lookback_limit"])
    f, sig = bot.build_signal_frame(df)

    # Build a gated signal array (copy sig; gate out entries)
    gated = sig.copy()
    if use_gate:
        gated[:] = 0
        for i in range(len(f)):
            s = int(sig.iloc[i])
            if s == 0: 
                continue
            row = {
                "open": float(f["Open"].iloc[i]), "high": float(f["High"].iloc[i]), "low": float(f["Low"].iloc[i]), "close": float(f["Close"].iloc[i]),
                "ema_fast": float(f["EMA_F"].iloc[i]), "ema_slow": float(f["EMA_S"].iloc[i]),
                "ema_fast_slope": float((f["EMA_F"].iloc[i] - f["EMA_F"].iloc[i-1]) if i>0 else 0.0),
                "ema_slow_slope": float((f["EMA_S"].iloc[i] - f["EMA_S"].iloc[i-1]) if i>0 else 0.0),
                "bb_width": float((f["Close"].rolling(20).std()*4 / (f["Close"].rolling(20).mean()+1e-12)).iloc[i]),
                "atr_pct": float(f["ATR_pct"].iloc[i]),
                "rsi": float(f["RSI"].iloc[i]),
                "signal": s,
            }
            if pass_gate(featurize_row(row)):
                gated.iloc[i] = s

    bt = bot.Backtester(f)
    res = bt.run(f, gated if use_gate else sig)
    return {
        "final_balance": res["final_balance"],
        "total_return": res["total_return"],
        "max_drawdown": res["max_drawdown"],
        "trades": int((gated if use_gate else sig).abs().sum()),
        "equity": res["equity"]
    }

if __name__ == "__main__":
    base = run_once(use_gate=False)
    gated = run_once(use_gate=True)

    summary = pd.DataFrame([
        {"variant": "BASE", **{k:v for k,v in base.items() if k != "equity"}},
        {"variant": "AI_GATE", **{k:v for k,v in gated.items() if k != "equity"}},
    ])
    summary.to_csv("bt_compare_summary.csv", index=False)

    # Save equity curves for plotting later
    base["equity"].to_csv("equity_base.csv", index=False)
    gated["equity"].to_csv("equity_ai_gate.csv", index=False)

    print(summary)
