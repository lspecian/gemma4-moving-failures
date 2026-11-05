"""Paper F6: is the public-leaderboard gap explained by best-of-n selection?

(1) From a downloaded public leaderboard (kaggle competitions leaderboard gemma-4-developer-agent -d): each team's best
    score by its submission count. (2) The best of n submissions expected for an agent with a fixed true pass rate p on
    the 58 public tasks (binomial; exact distribution of the maximum).

    python3 results/_analysis/paper/best_of_n.py <publicleaderboard.csv> [p=0.13]
"""

import csv
import math
import statistics
import sys

N_TASKS = 58


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1)) if k >= 0 else 0.0


def best_of(n, p, tasks=N_TASKS):
    """Mean and 10/50/90th percentiles of the best of n independent submissions, in tasks."""
    cdf = [binom_cdf(k, tasks, p) ** n for k in range(tasks + 1)]
    pmf = [cdf[0]] + [cdf[k] - cdf[k - 1] for k in range(1, tasks + 1)]
    q = lambda x: next(k for k in range(tasks + 1) if cdf[k] >= x)  # noqa: E731
    return sum(k * w for k, w in enumerate(pmf)), q(0.1), q(0.5), q(0.9)


def main():
    path = sys.argv[1]
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.13
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    teams = [(float(r['Score']), int(r['SubmissionCount'])) for r in rows if r['Score']]
    print(f'{len(teams)} teams with a score')
    bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 10 ** 6)]
    print('submissions  teams  median best  share >= 0.17')
    for lo, hi in bins:
        s = [sc for sc, n in teams if lo <= n <= hi]
        if s:
            lab = f'{lo}' if lo == hi else (f'{lo}-{hi}' if hi < 10 ** 6 else f'{lo}+')
            print(f'{lab:>11}  {len(s):5}  {statistics.median(s):11.2f}  {sum(x >= 0.17 for x in s) / len(s):12.0%}')
    print('best score  teams  median submissions')
    for sc in sorted({sc for sc, _ in teams}, reverse=True)[:8]:
        ns = [n for s, n in teams if s == sc]
        print(f'{sc:10.2f}  {len(ns):5}  {statistics.median(ns):18.1f}')
    sd = math.sqrt(N_TASKS * p * (1 - p))
    print(f'\ntrue rate {p:.2f} on {N_TASKS} tasks: one submission {N_TASKS * p:.1f} tasks, sd {sd:.1f} ({sd / N_TASKS:.3f})')
    print(' n  mean best  p10  p50  p90  (tasks; score = tasks/58)')
    for n in (1, 2, 4, 6, 8, 10, 15):
        m, a, b, c = best_of(n, p)
        print(f'{n:2}  {m:5.1f} ({m / N_TASKS:.2f})  {a:3}  {b:3}  {c:3}')


if __name__ == '__main__':
    main()
