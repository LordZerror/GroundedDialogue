"""Compute quadratic-weighted Cohen's kappa between human and LLM judge.

Inputs (long format, columns dialogue_id, turn_index, metric, score):
  eval/geval_scores.csv  — judge ratings (deepeval scores in [0,1])
  eval/human_ratings.csv — your ratings on a 1-5 integer scale

Judge scores are mapped 0-1 -> 1-5 to match the human scale before kappa.
"""
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


def deepeval_to_int(s):
    """Map deepeval's 0-1 score to 1-5 integer scale."""
    s = float(s)
    return max(1, min(5, round(s * 4) + 1))


def load_long(path, score_col="score", convert=None):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["dialogue_id"], int(r["turn_index"]), r["metric"])
            val = float(r[score_col])
            out[key] = convert(val) if convert else int(round(val))
    return out


def main():
    base = Path(__file__).parent
    judge = load_long(base / "geval_scores.csv", convert=deepeval_to_int)
    human = load_long(base / "human_ratings.csv")

    paired = defaultdict(lambda: ([], []))
    for key, h in human.items():
        if key in judge:
            metric = key[2]
            paired[metric][0].append(h)
            paired[metric][1].append(judge[key])

    print(f"{'Metric':<25} {'n':>4} {'kappa':>8}  interpretation")
    print("-" * 65)
    for metric, (h, j) in paired.items():
        if len(h) < 2:
            print(f"{metric:<25} {len(h):>4}    n/a   too few")
            continue
        k = cohen_kappa_score(h, j, weights="quadratic")
        if k >= 0.81:    interp = "almost perfect"
        elif k >= 0.61:  interp = "substantial (defensible)"
        elif k >= 0.41:  interp = "moderate"
        elif k >= 0.21:  interp = "fair"
        else:            interp = "slight"
        print(f"{metric:<25} {len(h):>4} {k:>8.3f}  {interp}")


if __name__ == "__main__":
    main()
