"""Pivot geval_scores.csv to (mode × metric) and report mean ± SD.

Output: results/geval_table.md — paste into the writeup with κ in caption.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def main():
    base = Path(__file__).parent
    scores_path = base / "geval_scores.csv"
    if not scores_path.exists():
        print("[generate_results] geval_scores.csv not found — run run_geval.py first.")
        return

    df = pd.read_csv(scores_path)
    pivot = df.groupby(["mode", "metric"])["score"].agg(["mean", "std", "count"])
    pivot = pivot.round(3)

    # Also compute overall per-metric
    overall = df.groupby("metric")["score"].agg(["mean", "std", "count"])
    overall.index = pd.MultiIndex.from_tuples(
        [("OVERALL", m) for m in overall.index], names=["mode", "metric"]
    )
    combined = pd.concat([pivot, overall.round(3)])

    out_dir = base.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "geval_table.md"

    with out_path.open("w") as f:
        f.write("# G-Eval Scores (mode × metric)\n\n")
        f.write(combined.to_markdown() + "\n")
        f.write("\n*κ values from `eval/kappa.py` should be cited in the caption.*\n")

    print(combined.to_markdown())
    print(f"\n[generate_results] wrote {out_path}")


if __name__ == "__main__":
    main()
