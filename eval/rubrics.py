"""Three G-Eval rubrics for tutoring-system dialogue evaluation.

Mode-Appropriateness is judgment-laden; No-Leak and Format are mechanical.
The judge model is set to a frontier model from a different family than
the local tutor server, controlling for self-preference bias


Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
"""
import os

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.models.llms import GeminiModel


def _make_judge():
    """Build the Gemini judge model.

    Reads the API key from GEMINI_API_KEY or GOOGLE_API_KEY.
    GEMINI_API_KEY takes precedence (deepeval looks for GOOGLE_API_KEY,
    so we copy it over if only GEMINI_API_KEY is set).
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = gemini_key
    return GeminiModel(model="gemini-2.5-flash")


def build_metrics():
    """Construct all three G-Eval metrics with the Gemini judge.

    Call this at runtime (not at import time) so the API key can be set
    before the model is constructed.
    """
    judge = _make_judge()

    ma = GEval(
        name="Mode Appropriateness",
        criteria=(
            "The assistant's response must fulfill the role specified for its "
            "current dialogue mode in a CS tutoring system."
        ),
        evaluation_steps=[
            "The assistant's mode is provided in the input as [mode: X].",
            "ONBOARDING_ACK: 1-2 sentences acknowledging topic + level; must NOT pose the MCQ itself.",
            "CORRECT_FIRST_TRY: briefly affirms correctness AND asks ONE open reflective question.",
            "CORRECT_AFTER_HINTS: acknowledges persistence and asks which hint helped.",
            "SCAFFOLD: gives a hint nudging toward the answer; must NOT state the correct option.",
            "EXPLAIN: explains the concept directly in 2-3 sentences; may reveal the answer.",
            "POST_EXPLAIN: invites reflection; must NOT re-explain; reminds student they can move on.",
            "REFLECTION: one short follow-up; reminds student they can say 'next question'.",
            "Score 1 if the response violates the role for its mode.",
            "Score 3 if it partially fulfills the role.",
            "Score 5 if it fulfills the role precisely.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.6,
    )

    nl = GEval(
        name="No Answer Leak",
        criteria=(
            "The assistant must not reveal the correct multiple-choice option "
            "(letter or verbatim option text) in modes where revealing the "
            "answer is forbidden."
        ),
        evaluation_steps=[
            "The mode and the correct answer are given in the input metadata.",
            "Forbidden modes: SCAFFOLD, ONBOARDING_ACK, REFLECTION.",
            "Permitted modes: EXPLAIN, POST_EXPLAIN, CORRECT_FIRST_TRY, CORRECT_AFTER_HINTS.",
            "Check whether the response contains the correct letter (e.g., 'D', 'option D', 'answer is D') or verbatim correct-option text.",
            "If the mode forbids leakage and the response leaks: score 1.",
            "If the mode forbids leakage and the response does not leak: score 5.",
            "If the mode permits leakage: score 5 unless the response is otherwise misleading.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.6,
    )

    fc = GEval(
        name="Format Compliance",
        criteria=(
            "The response must respect the length and format constraints stated "
            "in its system prompt (typically 1-2 or 2-3 sentences, conversational "
            "prose, no bullet points or numbered lists)."
        ),
        evaluation_steps=[
            "Count sentences in the response.",
            "Check for bullet points, numbered lists, or markdown headings.",
            "If the prompt specifies '1-2 sentences' and the response has more than 3, score 1-2.",
            "If bullets or lists appear when conversational prose was requested, deduct.",
            "Score 5 only if length and format match the prompt's specification exactly.",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.6,
    )

    return [ma, nl, fc]
