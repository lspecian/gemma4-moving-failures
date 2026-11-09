"""Kaggle Writeup card image (1280x640): title, subtitle and the handoff result (F4) as one bar series.

    uv run --with matplotlib python paper/figures/make_cover.py
"""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SERIES = '#2a78d6'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
SURFACE = '#fcfcfb'
OUT = pathlib.Path(__file__).with_name('cover.png')

fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=SURFACE)
fig.text(0.06, 0.80, 'Moving Failures Around', fontsize=40, fontweight='bold', color=INK, va='top')
fig.text(0.06, 0.66, 'A small coding agent acts on\ninformation it is shown, not on\ninstructions to find it',
         fontsize=19, color=INK2, va='top', linespacing=1.35)
fig.text(0.06, 0.12, 'Gemma 4 31B (w4a16) · 8 minutes per GitHub issue · ~6,000 episodes',
         fontsize=13, color=MUTED, va='bottom')

ax = fig.add_axes([0.66, 0.17, 0.29, 0.36], facecolor=SURFACE)
labels = ['own exploration, 8 min', 'own exploration, 30 min', "strong model's tool results",
          '+ its reasoning']
vals = [3.4, 17, 35, 44]
y = range(len(vals))[::-1]
ax.barh(list(y), vals, height=0.5, color=SERIES)
for yi, v in zip(y, vals):
    ax.text(v + 1, yi, f'{v:g}%', va='center', fontsize=12, color=INK)
ax.set_yticks(list(y), labels, fontsize=11, color=INK)
ax.set_xlim(0, 55)
ax.set_xticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
fig.text(0.53, 0.57, 'Gemma solves hard tasks when handed a\nstrong model\'s exploration',
         fontsize=12.5, color=INK2, va='bottom', linespacing=1.3)
fig.savefig(OUT, facecolor=SURFACE)
print(OUT)
