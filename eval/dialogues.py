"""Scripted dialogues for tutoring-system evaluation.

Each dialogue pins a known MCQ + a fixed sequence of student turns.
Pinning isolates dialogue management from question generation
(which is evaluated separately in tests/test_mcq_generator.py).

Each dialogue exercises one FSM path:
  1. correct_first_try     — ASSESSMENT → CORRECT_FIRST_TRY → REFLECTION
  2. wrong_then_correct    — ASSESSMENT → SCAFFOLD → CORRECT_AFTER_HINTS → REFLECTION
  3. three_wrongs_explain  — ASSESSMENT → SCAFFOLD x2 → EXPLAIN → POST_EXPLAIN
  4. clarification_mid_q   — mixed-initiative clarification path
  5. weird_onboarding      — paraphrased onboarding (no comma, no explicit level)
"""
import json

# The pinned MCQ. Correct answer is B; all dialogues reuse this.
SENTIMENT_MCQ = json.dumps({
    "skillLevel": "Apply",
    "question": "You have been provided with a raw dataset of customer support tickets containing mixed language (English and French) and varying sentiment intensities. Which of the following actions is the most appropriate next step to ensure the model successfully applies the 'sentiment analysis' framework to this specific input?",
    "options": {
        "A": "Manually translate every French ticket into English before feeding it to the model.",
        "B": "Preprocess the data by tokenizing the text and converting it to a numerical format suitable for the chosen algorithm.",
        "C": "Increase the size of the training dataset by five times to guarantee high accuracy.",
        "D": "Select a more complex neural network architecture than the one currently being used.",
    },
    "answer": "B",
})

DIALOGUES = [
    {
        "id": "correct_first_try",
        "label": "Student answers correctly on first attempt",
        "mcq": SENTIMENT_MCQ,
        "turns": [
            "sentiment analysis, intermediate",
            "B",
            "tokenization turns raw text into the numerical format the model needs",
            "next question",
        ],
    },
    {
        "id": "wrong_then_correct",
        "label": "One wrong answer, then correct after one scaffold hint",
        "mcq": SENTIMENT_MCQ,
        "turns": [
            "sentiment analysis, intermediate",
            "A",
            "B",
            "I see — manual translation doesn't convert text to numbers",
            "next question",
        ],
    },
    {
        "id": "three_wrongs_explain",
        "label": "Three wrong answers force the EXPLAIN path on the fourth attempt",
        "mcq": SENTIMENT_MCQ,
        "turns": [
            "sentiment analysis, intermediate",
            "A", # attempt 1 (Wrong) -> SOCRATIC
            "C", # attempt 2 (Wrong) -> CONTRAST
            "D", # attempt 3 (Wrong) -> WORKED_EXAMPLE
            "I have no idea", # attempt 4 (Wrong) -> EXPLAIN (forces answer)
            "okay I think I see now", # Reflection
            "next question",
        ],
    },
    {
        "id": "clarification_mid_q",
        "label": "Student asks a conceptual question mid-MCQ (mixed initiative)",
        "mcq": SENTIMENT_MCQ,
        "turns": [
            "sentiment analysis, intermediate",
            "what does tokenizing mean?",
            "B",
            "I get it now",
            "next",
        ],
    }
]
