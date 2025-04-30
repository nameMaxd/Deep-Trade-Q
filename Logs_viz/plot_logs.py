import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LOGFILE = 'train_finetune.log'
OUTDIR = './Logs_viz'
os.makedirs(OUTDIR, exist_ok=True)

# --- Парсинг логов ---
step_pattern = re.compile(r'Step (\d+): TrainProfit ([\-\d\.]+), ValProfit ([\-\d\.]+), Sharpe ([\-\d\.]+), AvgReward ([\-\d\.]+), TradesTrain ([\d\.]+), TradesVal ([\d\.]+)')
val_profit_pattern = re.compile(r'val\s+ep (\d+):profit ([\-\d\.]+)')

steps = []
val_profits = []
with open(LOGFILE, encoding='utf-8') as f:
    cur_step = None
    cur_sharpe = None
    cur_trades = None
    cur_val_profits = []
    for line in f:
        m1 = step_pattern.search(line)
        if m1:
            if cur_step is not None and cur_val_profits:
                steps.append({
                    'step': cur_step,
                    'sharpe': cur_sharpe,
                    'trades': cur_trades,
                    'val_profits': cur_val_profits
                })
            cur_step = int(m1.group(1))
            cur_sharpe = float(m1.group(4))
            cur_trades = float(m1.group(7))
            cur_val_profits = []
        m2 = val_profit_pattern.search(line)
        if m2:
            cur_val_profits.append(float(m2.group(2)))
    if cur_step is not None and cur_val_profits:
        steps.append({
            'step': cur_step,
            'sharpe': cur_sharpe,
            'trades': cur_trades,
            'val_profits': cur_val_profits
        })

# --- Подготовка данных для свечей ---
candles = []
for s in steps:
    vals = s['val_profits']
    if not vals:
        continue
    candle = {
        'step': s['step'],
        'open': vals[0],
        'high': max(vals),
        'low': min(vals),
        'close': vals[-1],
        'mean': np.mean(vals),
        'sharpe': s['sharpe'],
        'trades': s['trades']
    }
    candles.append(candle)

df = pd.DataFrame(candles)

# --- Построение графика ---
plt.figure(figsize=(16,8))

# Свечи профита
for idx, row in df.iterrows():
    color = 'g' if row['close'] >= row['open'] else 'r'
    plt.plot([row['step'], row['step']], [row['low'], row['high']], color=color, linewidth=2)
    plt.plot([row['step']-1000, row['step']], [row['open'], row['close']], color=color, linewidth=8, solid_capstyle='butt')

plt.xlabel('Step')
plt.ylabel('Val Profit (per episode)')
plt.title('Validation Profit Candles (box per 5000 steps)')

# Подписи сверху
for idx, row in df.iterrows():
    plt.text(row['step'], row['high']+max(df['high'])*0.02, f"Sharpe: {row['sharpe']:.2f}\nTrades: {row['trades']:.1f}",
             ha='center', va='bottom', fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'val_profit_candles.png'))
plt.close()

print('val_profit_candles.png сохранён в', OUTDIR)
