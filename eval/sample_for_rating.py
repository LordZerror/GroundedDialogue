"""Print 20 stratified turns to rate. Hide judge scores.

Outputs eval/to_rate.csv — open in a spreadsheet, add three columns:
  mode_appropriate_human, no_leak_human, format_human
Rate each on a 1-5 scale using the same rubric the judge has.

Then convert to long format and save as eval/human_ratings.csv with columns:
  dialogue_id, turn_index, metric, score
where metric must be one of:
  "Mode Appropriateness", "No Answer Leak", "Format Compliance"
(must match the metric names in geval_scores.csv exactly).
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVALUABLE_MODES = {
    "ONBOARDING_ACK", "CORRECT_FIRST_TRY", "CORRECT_AFTER_HINTS",
    "SCAFFOLD", "EXPLAIN", "POST_EXPLAIN", "REFLECTION",
}


def main():
    random.seed(42)
    base = Path(__file__).parent
    records = [json.loads(l) for l in open(base / "transcripts.jsonl")]

    by_mode = defaultdict(list)
    for r in records:
        if r["mode"] in EVALUABLE_MODES:
            by_mode[r["mode"]].append(r)

    sampled = []
    for mode, rs in by_mode.items():
        sampled.extend(random.sample(rs, min(4, len(rs))))

    out_path = base / "to_rate.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dialogue_id", "turn_index", "mode", "assistant_output"])
        for r in sampled:
            text = r["assistant_output"].replace("\n", " ")
            w.writerow([r["dialogue_id"], r["turn_index"], r["mode"], text])

    print(f"[sample_for_rating] wrote {len(sampled)} turns to {out_path}")
    print()
    print("Instructions:")
    print("  1. Open to_rate.csv in a spreadsheet.")
    print("  2. Add columns: mode_appropriate_human, no_leak_human, format_human")
    print("  3. Rate each on 1-5 using the same rubric as the judge.")
    print("  4. Save as eval/human_ratings.csv in long format with columns:")
    print('     dialogue_id, turn_index, metric, score')
    print('     where metric ∈ {"Mode Appropriateness", "No Answer Leak", "Format Compliance"}')


if __name__ == "__main__":
    main()
