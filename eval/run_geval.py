"""Evaluate harvested transcripts using G-Eval rubrics via direct Gemini API.

Bypasses deepeval's evaluate() to get full control over rate limiting and
retry logic. Implements G-Eval (Liu et al., EMNLP 2023) directly:
for each (turn, rubric), sends the criteria + evaluation steps to the
judge model and extracts a 1-5 score with reasoning.

Reads eval/transcripts.jsonl, writes eval/geval_scores.csv.
Requires GEMINI_API_KEY or GOOGLE_API_KEY in the environment.
"""
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .venv/.env
load_dotenv(Path(".venv/.env"))

# ── Model config ──────────────────────────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"
DELAY_BETWEEN_CALLS = 1           # seconds (Groq is very fast and has high rate limits)
MAX_RETRIES = 5
RETRY_BACKOFF = 5                 # seconds base backoff on 429/503


def _get_client():
    """Build Groq client from env vars."""
    import groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY in your environment.")
    return groq.Groq(api_key=key)


# ── Rubric definitions ────────────────────────────────────────────────────────

RUBRICS = {
    "Mode Appropriateness": {
        "criteria": (
            "The assistant's response must fulfill the role specified for its "
            "current dialogue mode in a CS tutoring system. The SCAFFOLD mode "
            "uses specific strategies (Socratic, Contrast, Worked Example)."
        ),
        "steps": [
            "The assistant's mode is provided in the input as [mode: X].",
            "ONBOARDING_ACK: 1-2 sentences acknowledging topic + level; must NOT pose the MCQ itself.",
            "CORRECT_FIRST_TRY: briefly affirms correctness AND asks ONE open reflective question.",
            "CORRECT_AFTER_HINTS: acknowledges persistence and asks which hint helped.",
            "SCAFFOLD: must ask a Socratic question, contrast a misunderstanding, or provide an analogous example. Must NEVER provide direct explanations or leak the answer.",
            "EXPLAIN: explains the concept directly in 2-3 sentences; may reveal the answer.",
            "POST_EXPLAIN: invites reflection; must NOT re-explain; reminds student they can move on.",
            "REFLECTION: one short follow-up; reminds student they can say 'next question'.",
            "Score 1 if the response violates the role for its mode.",
            "Score 3 if it partially fulfills the role.",
            "Score 5 if it fulfills the role precisely.",
        ],
    },
    "No Answer Leak": {
        "criteria": (
            "The assistant must not reveal the correct multiple-choice option "
            "(letter or verbatim option text) in modes where revealing the "
            "answer is forbidden."
        ),
        "steps": [
            "The mode and the correct answer are given in the input metadata.",
            "Forbidden modes: SCAFFOLD, ONBOARDING_ACK, REFLECTION.",
            "Permitted modes: EXPLAIN, POST_EXPLAIN, CORRECT_FIRST_TRY, CORRECT_AFTER_HINTS.",
            "Check whether the response contains the correct letter (e.g., 'D', 'option D', 'answer is D') or verbatim correct-option text.",
            "If the mode forbids leakage and the response leaks: score 1.",
            "If the mode forbids leakage and the response does not leak: score 5.",
            "If the mode permits leakage: score 5 unless the response is otherwise misleading.",
        ],
    },
    "Format Compliance": {
        "criteria": (
            "The response must respect the length and format constraints stated "
            "in its system prompt (typically 1-3 sentences, conversational "
            "prose, no bullet points or numbered lists)."
        ),
        "steps": [
            "Count sentences in the response.",
            "Check for bullet points, numbered lists, or markdown headings.",
            "If the response exceeds the 1-3 sentence limit, deduct heavily (score 1-2).",
            "If bullets or lists appear when conversational prose was requested, deduct.",
            "Score 5 only if length and format match the prompt's specification exactly.",
        ],
    },
}

EVALUABLE_MODES = {
    "ONBOARDING_ACK", "CORRECT_FIRST_TRY", "CORRECT_AFTER_HINTS",
    "SCAFFOLD", "EXPLAIN", "POST_EXPLAIN", "REFLECTION",
}


# ── Core evaluation ──────────────────────────────────────────────────────────

def build_judge_prompt(metric_name, rubric, input_text, actual_output):
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(rubric["steps"]))
    return f"""You are an expert evaluator for a CS tutoring dialogue system.

Criteria: {rubric['criteria']}

Evaluation steps:
{steps_text}

--- Input (metadata + conversation context) ---
{input_text}

--- Assistant response to evaluate ---
{actual_output}

Based on the criteria and evaluation steps above, provide:
1. A score from 1 to 5 (integer only).
2. A brief reason (1-2 sentences).

Respond in this exact format:
Score: <number>
Reason: <your reasoning>"""


def parse_score_and_reason(text):
    score_match = re.search(r"Score:\s*(\d)", text)
    reason_match = re.search(r"Reason:\s*(.+)", text, re.DOTALL)
    score = int(score_match.group(1)) if score_match else None
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    if score is not None:
        score = max(1, min(5, score))
    return score, reason


def call_judge(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "rate limit" in err_str.lower():
                wait = RETRY_BACKOFF * (attempt + 1)
                print(f"         ⏳ rate limited — retrying in {wait}s ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def load_transcripts(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_evaluable_turns(records):
    by_dialogue = {}
    for r in records:
        by_dialogue.setdefault(r["dialogue_id"], []).append(r)

    turns = []
    for dialogue_id, dialogue_turns in by_dialogue.items():
        history = []
        for t in dialogue_turns:
            if t["user_input"] is not None:
                history.append(f"USER: {t['user_input']}")
            if t["mode"] in EVALUABLE_MODES:
                context = "\n".join(history)
                input_text = (
                    f"[mode: {t['mode']}]\n"
                    f"[correct answer for current MCQ: D]\n"
                    f"[conversation so far]\n{context}"
                )
                turns.append({
                    "input_text": input_text,
                    "actual_output": t["assistant_output"],
                    "dialogue_id": dialogue_id,
                    "turn_index": t["turn_index"],
                    "mode": t["mode"],
                })
            history.append(f"ASSISTANT: {t['assistant_output']}")
    return turns


def main():
    base = Path(__file__).parent
    records = load_transcripts(base / "transcripts.jsonl")
    turns = build_evaluable_turns(records)
    total_evals = len(turns) * len(RUBRICS)
    est_min = (total_evals * DELAY_BETWEEN_CALLS) / 60
    print(f"[run_geval] {len(turns)} turns × {len(RUBRICS)} rubrics = {total_evals} evals")
    print(f"[run_geval] ~{est_min:.0f} min estimated (rate-limited)")

    client = _get_client()
    rows = []

    for i, turn in enumerate(turns):
        label = f"{turn['dialogue_id']} t{turn['turn_index']} [{turn['mode']}]"
        print(f"\n({i+1}/{len(turns)}) {label}")

        for metric_name, rubric in RUBRICS.items():
            prompt = build_judge_prompt(metric_name, rubric, turn["input_text"], turn["actual_output"])
            response_text = call_judge(client, prompt)
            score, reason = parse_score_and_reason(response_text)

            print(f"  {metric_name}: {score}")
            rows.append({
                "dialogue_id": turn["dialogue_id"],
                "turn_index": turn["turn_index"],
                "mode": turn["mode"],
                "metric": metric_name,
                "score": score,
                "reason": reason.replace("\n", " "),
            })
            time.sleep(DELAY_BETWEEN_CALLS)

    out = base / "geval_scores.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dialogue_id", "turn_index", "mode", "metric", "score", "reason"])
        for row in rows:
            w.writerow([row["dialogue_id"], row["turn_index"], row["mode"],
                        row["metric"], row["score"], row["reason"]])

    print(f"\n✅ wrote {out} ({len(rows)} scores)")
    from collections import defaultdict
    by_metric = defaultdict(list)
    for r in rows:
        if r["score"] is not None:
            by_metric[r["metric"]].append(r["score"])
    for metric, scores in by_metric.items():
        print(f"  {metric}: mean={sum(scores)/len(scores):.2f} (n={len(scores)})")


if __name__ == "__main__":
    main()
