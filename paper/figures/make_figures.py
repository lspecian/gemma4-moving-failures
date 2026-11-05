"""Figures for paper/draft.md. Numbers from experiments/LOG.md (experiment ids in the comments).

    python paper/figures/make_figures.py      # needs matplotlib; writes fig1_strong_model.{png,svg}, fig2_handoff.{png,svg}
"""

import pathlib

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
SERIES = '#2a78d6'      # one series, one color (categorical slot 1 of the reference palette)
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
GRID, SURFACE = '#e6e5e1', '#fcfcfb'


def bars(ax, labels, values, vmax, ref=None, ref_label=None, fmt='{}'):
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, height=0.42, color=SERIES, zorder=3)
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.012, yi, (f'{v:.1f}%' if v < 10 else f'{v:.0f}%') if fmt == 'pct' else fmt.format(v), va='center', ha='left', fontsize=10, color=INK, zorder=4)
    ax.set_yticks(y, labels, fontsize=10, color=INK)
    ax.set_xlim(0, vmax * 1.08)
    ax.tick_params(axis='x', colors=INK2, labelsize=9, length=0)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.set_facecolor(SURFACE)
    if ref is not None:
        ax.axvline(ref, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.text(ref - vmax * 0.012, len(labels) - 1, ref_label, ha='right', va='center', fontsize=9, color=INK2)


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def fig1():
    # per-attempt success on the 23 eval113 tasks solved by some DeepSeek run and by none of four Gemma runs
    # (E069, E070, E073, E075); baseline over all 14 Gemma real-budget runs (results/_analysis/paper/baselines.py)
    labels = ['Gemma alone, 8 min (14 runs, 296 attempts)',              # 10/296 = 3.4%
              'Gemma alone, 30 min / 100 calls (23)',                     # E092: 4/23 = 17%
              "Gemma after the strong model's\ntool calls and results (34)",   # E090/E091: 12/34 = 35%
              '... and its reasoning (34)']                               # E088/E089: 15/34 = 44%
    values = [3.4, 17.4, 35.3, 44.1]
    fig, ax = plt.subplots(figsize=(7.2, 2.9), facecolor=SURFACE)
    bars(ax, labels, values, 50, fmt='pct')
    ax.set_xlabel('share of attempts that pass, on 23 tasks a strong model solves (attempts in brackets)', fontsize=9.5,
                  color=INK2)
    fig.suptitle("Given a strong model's exploration, Gemma's attempts succeed ten times as often", x=0.02, ha='left',
                 fontsize=12, color=INK, fontweight='bold')
    save(fig, 'fig2_handoff')


def fig2():
    labels = ['Gemma 4 31B, 8 min (E075)', 'DeepSeek V4 Pro, 8 min (E082)', 'DeepSeek V4.1 Flash, 8 min (E083)',
              'V4.1 Flash, 30 min, our tools (E084)', 'V4.1 Flash, 30 min, stock tools (E085)']
    values = [45, 49, 50, 67, 75]
    fig, ax = plt.subplots(figsize=(7.2, 3.1), facecolor=SURFACE)
    bars(ax, labels, values, 113, ref=113, ref_label='all 113')
    ax.set_xlabel('tasks solved, of 113 (8 min = 28 calls at 35 tok/s; 30 min = 100 calls, unpaced)', fontsize=9.5,
                  color=INK2)
    fig.suptitle('A stronger model in our harness: little gain at our budget, much more with time', x=0.02, ha='left', fontsize=12,
                 color=INK, fontweight='bold')
    save(fig, 'fig1_strong_model')


if __name__ == '__main__':
    fig1()
    fig2()
    print('wrote', sorted(p.name for p in OUT.glob('fig*')))
