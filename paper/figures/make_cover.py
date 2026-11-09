"""Kaggle Writeup card and thumbnail image (1120x560, 2x Kaggle's 560x280 card).

Kaggle crops the thumbnail as the centre square of the card, so everything readable sits in the middle half:
the title on three lines and a short subtitle. The handoff result (F4) is drawn as faint bars in the margins,
which only the card shows.

    uv run --with matplotlib python paper/figures/make_cover.py
"""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SERIES = '#2a78d6'
INK, INK2 = '#0b0b0b', '#52514e'
SURFACE = '#fcfcfb'
OUT = pathlib.Path(__file__).with_name('cover.png')

fig = plt.figure(figsize=(11.2, 5.6), dpi=100, facecolor=SURFACE)
fig.text(0.5, 0.80, 'Moving\nFailures\nAround', fontsize=54, fontweight='bold', color=INK,
         ha='center', va='top', linespacing=1.05)
fig.text(0.5, 0.22, 'Gemma 4 acts on what it is shown,\nnot on instructions to find it',
         fontsize=17, color=INK2, ha='center', va='top', linespacing=1.3)

# margins: the handoff solve rates (3.4, 17, 35, 44 %) as bars growing toward the centre
ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
for i, v in enumerate([3.4, 17, 35, 44]):
    w = 0.19 * v / 44
    y = 0.62 - i * 0.12
    ax.add_patch(Rectangle((0.0, y), w, 0.06, color=SERIES, alpha=0.85))
    ax.add_patch(Rectangle((1.0 - w, y), w, 0.06, color=SERIES, alpha=0.85))
fig.savefig(OUT, facecolor=SURFACE)
print(OUT)
