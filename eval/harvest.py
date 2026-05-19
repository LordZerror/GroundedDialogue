"""Run scripted dialogues through MisconceptionTutoringSystem with
retrieveQuestion patched. Record the system-prompt mode that produced
each assistant reply.

Output: eval/transcripts.jsonl (one JSON line per assistant turn).
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Make the project root importable when running from eval/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tutoringSystem
from tutoringSystem import MisconceptionTutoringSystem, _SYSTEM_PROMPTS
from eval.dialogues import DIALOGUES

# Reverse the prompt -> mode map so we can label replies by which system
# prompt was last in effect when chat() was called.
_PROMPT_TO_MODE = {v: k for k, v in _SYSTEM_PROMPTS.items()}


def _wrap_chat(last_mode_box):
    """Return a chat() wrapper that records the most recent mode key used."""
    original = tutoringSystem.chat

    def tracking_chat(session_id, prompt, system_prompt=None, reset=False):
        if system_prompt and system_prompt in _PROMPT_TO_MODE:
            last_mode_box["value"] = _PROMPT_TO_MODE[system_prompt]
        return original(session_id, prompt, system_prompt=system_prompt, reset=reset)

    return tracking_chat


def run_dialogue(dialogue):
    """Run one dialogue, return list of turn records."""
    last_mode_box = {"value": None}
    records = []

    with patch.object(tutoringSystem, "retrieveQuestion", return_value=dialogue["mcq"]), \
         patch.object(tutoringSystem, "chat", _wrap_chat(last_mode_box)):

        sys_obj = MisconceptionTutoringSystem()
        session_id, greeting = sys_obj.start_conversation()

        # Turn 0: the static greeting; not evaluated by GEval but logged.
        records.append({
            "dialogue_id": dialogue["id"],
            "turn_index": 0,
            "user_input": None,
            "assistant_output": greeting,
            "mode": "GREETING",
            "scaffold_level": None,
            "attempts": 0,
        })

        for i, student_turn in enumerate(dialogue["turns"], start=1):
            last_mode_box["value"] = None
            reply = sys_obj.process_user_input(session_id, student_turn)
            session = sys_obj.sessions[session_id]
            records.append({
                "dialogue_id": dialogue["id"],
                "turn_index": i,
                "user_input": student_turn,
                "assistant_output": reply,
                # Prefer the prompt-derived mode; fall back to session.mode.
                "mode": last_mode_box["value"] or session.mode,
                "scaffold_level": getattr(session, "scaffold_level", None),
                "attempts": getattr(session, "attempts", None),
            })
    return records


def main():
    out_path = Path(__file__).parent / "transcripts.jsonl"
    with out_path.open("w") as f:
        for dialogue in DIALOGUES:
            print(f"[harvest] running {dialogue['id']}")
            for record in run_dialogue(dialogue):
                f.write(json.dumps(record) + "\n")
    print(f"[harvest] wrote {out_path}")


if __name__ == "__main__":
    main()
