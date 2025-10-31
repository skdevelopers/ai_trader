#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools
import argparse
import pandas as pd
from copy import deepcopy

import ai_trading_bot as bot

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

SWEEP_LIMIT = int(os.getenv("SWEEP_LIMIT", "2000"))

def ensure_base_data() -> pd.DataFrame:
    if bot.CONFIG["backtest_csv"]:
        df = pd.read_csv(bot.CONFIG["backtest_csv"])
        df = bot.normalize_ohlcv(df)
        return df.tail(SWEEP_LIMIT).copy()

    cache_path = os.path.join(bot.CONFIG["cache_dir"], f'{bot.CONFIG["symbol"]}_{bot.CONFIG["interval"]}_sweep.csv')
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            df = bot.normalize_ohlcv(df)
            return df.tail(SWEEP_LIMIT).copy()
        except Exception:
            pass

    df = bot.get_klines(bot.CONFIG["symbol"], bot.CONFIG["interval"], max(SWEEP_LIMIT, bot.CONFIG["lookback_limit"]))
    df.to_csv(cache_path, index=False)
    return df.tail(SWEEP_LIMIT).copy()

def run_once_with_progress(df: pd.DataFrame, overrides: dict, outer_bar=None) -> dict:
    """
    Runs a single backtest with per-bar callback that updates the outer tqdm.
    """
    original = {}
    for k, v in overrides.items():
        original[k] = bot.CONFIG.get(k)
        bot.CONFIG[k] = v

    try:
        f = bot.normalize_ohlcv(df)
        f["EMA_F"] = bot.ema(f["Close"], int(bot.CONFIG["ema_fast"]))
        f["EMA_S"] = bot.ema(f["Close"], int(bot.CONFIG["ema_slow"]))
        f["RSI"]   = bot.rsi(f["Close"], int(bot.CONFIG["rsi_period"]))
        f["ATR"]   = bot.atr(f, int(bot.CONFIG["atr_period"]))
        f["ATR_pct"] = f["ATR"] / f["Close"] * 100
        raw = bot.detect_patterns(f)
        sig = bot.apply_filters(f, raw)

        # Disable internal tqdm (we’ll drive progress via callback)
        bt = bot.Backtester(f, show_progress=False)

        def on_bar(done: int, total: int):
            # Use a small fraction so each combo contributes ~1.0 to outer bar when complete.
            if outer_bar is not None:
                # advance proportionally; cap to avoid over-advance due to rounding
                outer_bar.update(0)  # ensure refresh
                # no-op here; we advance per-bar by a precomputed quantum in caller

                # We could call set_postfix here if you want live PnL:
                # if done % 200 == 0:
                #     outer_bar.set_postfix_str(f"bars {done}/{total}")

                pass

        res = bt.run(f, sig, on_bar=on_bar)

        return {
            **overrides,
            "final_balance": res["final_balance"],
            "total_return": res["total_return"],
            "max_drawdown": res["max_drawdown"],
            "bars": len(f),
        }
    finally:
        for k in overrides.keys():
            bot.CONFIG[k] = original[k]

def parse_args():
    p = argparse.ArgumentParser(description="Parameter Sweep")
    p.add_argument("--progress", action="store_true", help="Show a progress bar over parameter combos (and per-bar inside each)")
    p.add_argument("--limit", type=int, default=0, help="Limit number of combos to run (0 = all)")
    p.add_argument("--outfile", type=str, default="param_results_5m.csv", help="CSV output path")
    p.add_argument("--heartbeat", type=int, default=int(os.getenv("SWEEP_HEARTBEAT_EVERY", "25")),
                   help="Print a heartbeat every N combos when not using tqdm")
    return p.parse_args()

if __name__ == "__main__":
    # Force backtest mode for sweeps
    bot.CONFIG["mode"] = "BACKTEST"

    args = parse_args()

    base = deepcopy(bot.CONFIG)
    base.update({
        "interval": "5m",
        "risk_per_trade": 0.001,
        "max_daily_loss": 0.01,
        "fees_bp": 10.0,
        "slippage_bp": 7.0,
        "ema_fast": 21,
        "ema_slow": 55,
        "min_atr_pct": 0.10,
        "max_atr_pct": 1.20,
        "atr_stop_k": 2.2,
        "atr_trail_k": 1.0,
        "tp1_R": 0.8,
        "tp2_R": 1.6,
        "partial_tp1": 0.6,
    })
    for k, v in base.items():
        bot.CONFIG[k] = v

    df = ensure_base_data()

    grids = {
        "ema_fast": [20, 21, 25],
        "ema_slow": [50, 55, 60],
        "min_atr_pct": [0.08, 0.10, 0.12],
        "max_atr_pct": [1.0, 1.2],
        "atr_stop_k": [2.0, 2.2, 2.4],
        "atr_trail_k": [0.9, 1.0, 1.2],
        "tp1_R": [0.7, 0.8, 0.9],
        "tp2_R": [1.4, 1.6, 1.8],
        "partial_tp1": [0.5, 0.6, 0.7],
        "risk_per_trade": [0.001, 0.0015, 0.002],
    }

    keys = list(grids.keys())
    combos = list(itertools.product(*[grids[k] for k in keys]))

    if args.limit and args.limit > 0:
        combos = combos[:args.limit]

    rows = []
    try:
        if args.progress and _HAS_TQDM:
            # Outer bar spans all combos; we’ll advance by ~1 per finished combo,
            # plus small increments per bar to show life during long runs.
            outer = tqdm(total=len(combos), desc="Param Sweep (combos)", unit="combo")
            for idx, vals in enumerate(combos):
                overrides = dict(zip(keys, vals))

                # tiny per-bar increments: 1.0 / (len(combos) * bars_per_run)
                # We don’t know bars_per_run exactly; approximate as len(df)
                per_bar_increment = 1.0 / max(1, len(combos) * len(df))
                bars_done_before = 0

                def per_bar_tick(done: int, total: int):
                    nonlocal bars_done_before
                    inc = (done - bars_done_before) * per_bar_increment
                    bars_done_before = done
                    # update outer bar in tiny steps; tqdm handles fractional updates
                    outer.update(inc)

                # run with a wrapper that injects per_bar updates via Backtester.run callback
                # We pass outer but hook per_bar_tick by monkeypatching run call
                # (Simplest: call our run_once_with_progress and rely on internal callback to call per_bar_tick via closure.)
                # For that, we need to temporarily wrap bot.Backtester.run; easier is to just call run_once_with_progress
                # and let it call Backtester with a callback that references outer via closure variable.
                # So redefine a small inner that binds per_bar_tick:

                # Patch: pass outer to run function and tick in on_bar
                def run_bound(df_local, ov_local):
                    original_run = bot.Backtester.run
                    def run_with_cb(self_bt, df_bt, sig_bt, on_bar=None):
                        # inject our cb that uses per_bar_tick
                        def cb(done, total):
                            per_bar_tick(done, total)
                            if on_bar:
                                on_bar(done, total)
                        return original_run(self_bt, df_bt, sig_bt, on_bar=cb)
                    bot.Backtester.run = run_with_cb
                    try:
                        return run_once_with_progress(df_local, ov_local, outer_bar=outer)
                    finally:
                        bot.Backtester.run = original_run

                r = run_bound(df, overrides)
                rows.append(r)
                # ensure at least 1.0 per finished combo
                outer.update(max(0.0, 1.0 - (bars_done_before * per_bar_increment)))
            outer.close()
        else:
            heartbeat_every = max(1, int(args.heartbeat))
            for idx, vals in enumerate(combos):
                overrides = dict(zip(keys, vals))
                if idx % heartbeat_every == 0:
                    print(f"[sweep] {idx}/{len(combos)} combos...", flush=True)
                r = run_once_with_progress(df, overrides, outer_bar=None)
                rows.append(r)
    except KeyboardInterrupt:
        print("\nInterrupted! Writing partial results...", flush=True)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[out["max_drawdown"] <= 0.08].sort_values(["total_return"], ascending=False)
        out.to_csv(args.outfile, index=False)
        print(f"Saved {args.outfile} (top 10):")
        print(out.head(10))
    else:
        print("No results to save.")
    print("DONE.")
