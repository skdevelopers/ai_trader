AI Trading Bot (MVP)
- Pattern+Indicator confluence (Engulfing, Hammer, Shooting Star) + EMA/RSI/ATR
- Risk controls: per-trade risk %, ATR SL, partial TP(1R), ATR trailing, daily loss stop.
- Modes: BACKTEST and LIVE (Binance Testnet).

Usage:
  pip install pandas numpy requests python-dotenv
  # Set envs via your shell or a .env loader (not included here)
  # activate virtual env 
  .\.venv\Scripts\activate on Linux source .venv/bin/activate
  python ai_trading_bot.py
  
python ai_trading_bot.py --progress --plot
# outputs:
# - live tqdm bar (or heartbeat logs if tqdm missing)
# - equity_curve.csv
# - backtest_summary.json
# - equity_curve.png
# - prints "DONE." at end

python param_sweep.py --progress --limit 50 --outfile param_results_5m.csv
# shows a progress bar across 50 combos, writes CSV, prints "DONE."
