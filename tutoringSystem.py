from __future__ import annotations

import json
import re

from session import DialogueSession
from llmHelpers import chat, retrieveQuestion
from scaffolding import OntologyService, ScaffoldingEngine


# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_PROMPTS: dict[str, str] = {
    "ONBOARDING": (
        "You are a helpful assistant. Extract a CS topic and proficiency level "
        "from the student's message. "
        "Respond ONLY with a valid JSON object — no prose, no markdown fences — "
        'like: {"topic": "machine learning", "proficiency": "beginner"}. '
        "proficiency must be exactly one of: beginner, intermediate, proficient. "
        "If proficiency is not mentioned default to beginner."
    ),
    # Warm acknowledgment after onboarding is parsed, before the first question appears.
    "ONBOARDING_ACK": (
        "You are a warm, encouraging CS tutor. "
        "The student has just told you their topic and proficiency level. "
        "Write 1-2 natural, conversational sentences that acknowledge their choice "
        "and let them know you're about to give them a question. "
        "Do NOT generate the question itself — just the bridging remark."
    ),
    # Correct on the very first attempt — praise + one open reflection question.
    "CORRECT_FIRST_TRY": (
        "You are an encouraging CS tutor. "
        "The student just answered a multiple-choice question correctly on their first try. "
        "Write 1-2 sentences: briefly acknowledge that they got it right (you may name "
        "the correct answer), then ask ONE open reflective question that invites them to "
        "articulate *why* that answer is correct. "
        "Keep it conversational — no bullet points."
    ),
    # Correct after one or more scaffold hints — acknowledge persistence.
    "CORRECT_AFTER_HINTS": (
        "You are an encouraging CS tutor. "
        "The student has just answered correctly after receiving hints. "
        "Write 1-2 sentences: warmly acknowledge their persistence in working through "
        "the hints, then ask which part of the explanation helped them see the answer. "
        "Keep it conversational — no bullet points."
    ),
    # After the forced EXPLAIN path — invite reflection, do NOT re-explain.
    "POST_EXPLAIN": (
        "You are a reflective CS tutor. "
        "The student has just been walked through the correct answer after several wrong attempts. "
        "Write 1-2 sentences that gently invite them to reflect on what they now understand "
        "differently. Do NOT re-explain the concept. "
        "End by reminding them they can type 'next question' when they are ready to continue."
    ),
    "REFLECTION": (
        "You are a reflective CS tutor. "
        "Ask ONE short follow-up question to consolidate the student's learning. "
        "At the end, remind them they can say things like 'next question', "
        "'I'm done', or 'move on' whenever they want a new question. "
        "Keep it under three sentences."
    ),
    # Intent classifier — returns only the word NEXT or CONTINUE.
    "NEXT_Q_INTENT": (
        "You are an intent classifier for a tutoring system. "
        "The student is in the reflection phase after answering a question. "
        "Decide whether their message expresses a desire to move on to a new question "
        "(NEXT) or to continue the current reflection (CONTINUE). "
        "Respond with ONLY the single word NEXT or CONTINUE — nothing else."
    ),
    "EXPLAIN": (
        "You are a clear, concise CS tutor. "
        "Explain the concept directly and simply in 2-3 sentences. "
        "You may reveal the correct answer."
    ),
}


def _sys(mode: str) -> str:
    return _SYSTEM_PROMPTS.get(mode, "You are a helpful CS tutoring assistant.")


# ── Answer-extraction helpers ─────────────────────────────────────────────────

# Matches: A  a  A)  a)  A.  A:  (A)  A) some text
_LETTER_RE = re.compile(r"^\s*\(?([A-Da-d])\)?[.):\s]?")

# Question patterns that trigger mixed-initiative clarification
_CLARIFY_RE = re.compile(
    r"^(why\b"
    r"|what\s+(is|are|does|do|means?)\b"
    r"|how\s+(does|do|is|are|can)\b"
    r"|can\s+you\s+explain\b"
    r"|explain\b"
    r"|i\s+don'?t\s+understand\b"
    r"|i'?m\s+confused\b)",
    re.IGNORECASE,
)

def _is_next_q(text: str, session_id: str) -> bool:
    """
    Use an LLM to detect whether the student wants to move to a new question.
    This handles any natural phrasing ("I'm done", "move on", "understood, next",
    "can we do another one", etc.) rather than a fixed keyword list.
    Falls back to False on any error so the session never crashes.
    """
    try:
        result = chat(
            session_id + "_intent",
            f'Student message: "{text}"',
            system_prompt=_SYSTEM_PROMPTS["NEXT_Q_INTENT"],
        )
        return result.strip().upper().startswith("NEXT")
    except Exception:
        return False


def _extract_letter(text: str) -> str | None:
    """Return uppercase A/B/C/D if the input begins with an answer letter; else None."""
    m = _LETTER_RE.match(text)
    return m.group(1).upper() if m else None


def _is_clarification(text: str) -> bool:
    """Return True when the student is asking a conceptual question mid-exercise."""
    return bool(_CLARIFY_RE.match(text.strip()))


# ── Tutoring system ───────────────────────────────────────────────────────────

class MisconceptionTutoringSystem:

    def __init__(self):
        self.sessions: dict[str, DialogueSession] = {}
        self.ontology = OntologyService()
        self.scaffold = ScaffoldingEngine(self.ontology, chat)

    # ── Public API ────────────────────────────────────────────────────────────

    def start_conversation(self) -> tuple[str, str]:
        session = DialogueSession()
        self.sessions[session.session_id] = session
        print(f"[state] {session.mode}")
        return session.session_id, (
            "Hi! I'm your CS tutor.\n"
            "What topic would you like to practise? "
            "Also tell me your proficiency level: beginner, intermediate, or proficient."
        )

    def process_user_input(self, session_id: str, user_input: str) -> str:
        session = self.sessions[session_id]
        print(f"[state] {session.mode}  |  attempts={session.attempts}")

        # ── Mixed initiative: clarification questions in ASSESSMENT / SCAFFOLD ──
        if session.mode in ("ASSESSMENT", "SCAFFOLD") and _is_clarification(user_input):
            return self._clarify_and_return(session, user_input)

        dispatch = {
            "ONBOARDING":  self._handle_onboarding,
            "ASSESSMENT":  self._handle_assessment,
            "SCAFFOLD":    self._handle_scaffold,
            "EXPLAIN":     self._handle_explain,
            "REFLECTION":  self._handle_reflection,
        }
        handler = dispatch.get(session.mode)
        if handler is None:
            return "Something went wrong. Type 'next question' to continue."
        return handler(session, user_input)

    # ── Mixed-initiative clarification ────────────────────────────────────────

    def _clarify_and_return(self, session: DialogueSession, user_input: str) -> str:
        """
        Answer the student's conceptual question, then return to the same state
        so the exercise can continue without the attempt counter changing.
        """
        saved_mode = session.mode           # will be ASSESSMENT or SCAFFOLD
        prompt = (
            f"Topic: {session.topic}\n"
            f"Current question: {session.current_question or 'N/A'}\n"
            f"Student asks: {user_input}\n\n"
            "Answer their question in 2-3 sentences, then invite them to "
            "return to the exercise (remind them of the answer choices if helpful)."
        )
        reply = chat(session.session_id, prompt, system_prompt=_sys("EXPLAIN"))
        session.mode = saved_mode           # restore — no state change
        return reply

    # ── Onboarding ────────────────────────────────────────────────────────────

    def _handle_onboarding(self, session: DialogueSession, user_input: str) -> str:
        """Use the LLM to extract topic + proficiency; fall back to comma split."""
        raw = chat(
            session.session_id + "_onboard",
            f'Extract topic and proficiency from: "{user_input}"',
            system_prompt=_sys("ONBOARDING"),
        )
        try:
            data = json.loads(re.sub(r"```json|```", "", raw).strip())
            topic = str(data.get("topic", "")).strip()
            prof  = str(data.get("proficiency", "beginner")).strip().lower()
        except (json.JSONDecodeError, AttributeError, TypeError):
            # Graceful fallback: comma-split
            parts = [p.strip() for p in user_input.split(",")]
            topic = parts[0].strip()
            prof  = parts[1].lower().strip() if len(parts) > 1 else "beginner"

        if not topic:
            return (
                "I didn't catch the topic. Please tell me which CS topic "
                "you'd like to practise and your level "
                "(beginner / intermediate / proficient)."
            )
        if prof not in {"beginner", "intermediate", "proficient"}:
            prof = "beginner"

        session.topic       = topic
        session.proficiency = prof.title()   # "Beginner" / "Intermediate" / "Proficient"

        # ── LLM bridge: acknowledge the topic/level before showing the question ──
        ack = chat(
            session.session_id,
            (
                f"Topic chosen: {session.topic}\n"
                f"Proficiency level: {session.proficiency}\n\n"
                "Write a short bridging remark acknowledging their choice."
            ),
            system_prompt=_sys("ONBOARDING_ACK"),
        )

        question_block = self._load_new_question(session)
        return f"{ack}\n\n{question_block}"

    # ── Question loading ──────────────────────────────────────────────────────

    def _load_new_question(self, session: DialogueSession) -> str:
        try:
            mcq = json.loads(retrieveQuestion(session.topic, session.proficiency.lower()))
        except (json.JSONDecodeError, Exception) as exc:
            print(f"[error] question generation failed: {exc}")
            return (
                "I had trouble generating a question right now. "
                "Type 'next question' to try again, or name a different topic."
            )

        correct_opt = mcq["answer"].strip().upper()
        opts        = mcq["options"]          # {"A": "...", "B": "...", ...}

        session.bloom_level       = mcq["skillLevel"]
        session.question_concept  = session.topic   # used by ScaffoldingEngine
        session.current_question  = mcq["question"]
        session.options           = dict(opts)
        session.correct_option    = correct_opt
        session.correct_answer    = opts[correct_opt]
        session.attempts          = 0
        session.scaffold_level    = None
        session.scaffold_strategy = None
        session.concepts_used     = []
        session.mode              = "ASSESSMENT"

        return self._render_question(mcq["question"], opts, mcq["skillLevel"])

    @staticmethod
    def _render_question(question: str, opts: dict, bloom: str) -> str:
        return (
            f"[Bloom: {bloom}]\n"
            f"{question}\n"
            f"  A) {opts['A']}\n"
            f"  B) {opts['B']}\n"
            f"  C) {opts['C']}\n"
            f"  D) {opts['D']}\n"
        )

    def _redisplay_question(self, session: DialogueSession) -> str:
        """Re-render the current question using stored option texts."""
        return (
            f"{session.current_question}\n"
            f"  A) {session.options.get('A', '')}\n"
            f"  B) {session.options.get('B', '')}\n"
            f"  C) {session.options.get('C', '')}\n"
            f"  D) {session.options.get('D', '')}\n"
            "Please answer with A, B, C, or D."
        )

    # ── ASSESSMENT ────────────────────────────────────────────────────────────

    def _handle_assessment(self, session: DialogueSession, user_input: str) -> str:
        """
        Accept the student's first answer.
        - Unrecognised input  → re-display question (no attempt consumed)
        - Correct             → LLM praise + open reflection question → REFLECTION
        - Wrong               → increment attempts, enter SCAFFOLD with first hint
        """
        letter = _extract_letter(user_input)
        if letter is None:
            return self._redisplay_question(session)

        if letter == session.correct_option:
            session.mode = "REFLECTION"
            # ── LLM: contextual praise referencing the specific question ──────
            return chat(
                session.session_id,
                (
                    f"Topic: {session.topic}\n"
                    f"Bloom level: {session.bloom_level}\n"
                    f"Question: {session.current_question}\n"
                    f"Correct answer: {session.correct_option}) {session.correct_answer}\n"
                    f"Student answered: {letter}\n\n"
                    "Acknowledge they got it right and ask one reflective question."
                ),
                system_prompt=_sys("CORRECT_FIRST_TRY"),
            )

        # Wrong first answer
        session.attempts += 1          # attempts = 1
        session.mode = "SCAFFOLD"
        return self._generate_scaffold_reply(session, user_input)

    # ── SCAFFOLD ──────────────────────────────────────────────────────────────

    def _handle_scaffold(self, session: DialogueSession, user_input: str) -> str:
        """
        Loop here while the student is getting hints.
        - Correct answer at any point → LLM praise referencing the journey → REFLECTION
        - Wrong / non-letter          → increment attempts; escalate strategy or
                                        fall through to forced explanation
        Attempt → Strategy mapping (via ScaffoldingEngine.choose_strategy):
            1 → SOCRATIC
            2 → CONTRAST
            3 → WORKED_EXAMPLE
            4+ → EXPLAIN (forced)
        """
        letter = _extract_letter(user_input)

        # Correct answer detected
        if letter is not None and letter == session.correct_option:
            session.mode = "REFLECTION"
            # ── LLM: acknowledge persistence + ask what clicked ───────────────
            return chat(
                session.session_id,
                (
                    f"Topic: {session.topic}\n"
                    f"Bloom level: {session.bloom_level}\n"
                    f"Question: {session.current_question}\n"
                    f"Correct answer: {session.correct_option}) {session.correct_answer}\n"
                    f"Attempts before getting it right: {session.attempts}\n"
                    f"Last scaffold strategy used: {session.scaffold_strategy or 'hint'}\n\n"
                    "Acknowledge their persistence and ask which part of the hint clicked."
                ),
                system_prompt=_sys("CORRECT_AFTER_HINTS"),
            )

        # Wrong answer (or non-letter input treated as wrong)
        session.attempts += 1

        if session.attempts >= 4:
            return self._force_explain(session, user_input)

        return self._generate_scaffold_reply(session, user_input)

    def _generate_scaffold_reply(self, session: DialogueSession, user_input: str) -> str:
        """
        Delegate to ScaffoldingEngine:
          1. Infers misunderstanding via LLM
          2. Fetches CSO ancestor chain at the appropriate depth (LOW/MEDIUM/HIGH)
          3. Generates a hint grounded in the ontology context

        The answer choices are appended after the hint so the student always
        has context on what to answer next. Mode stays SCAFFOLD.
        """
        response = self.scaffold.generate_support(session, user_input)
        session.scaffold_level    = response.context.scaffold_level
        session.scaffold_strategy = response.strategy
        session.concepts_used     = [n.label for n in response.context.concept_chain]

        # Re-surface the answer choices so the student can respond immediately
        choices = (
            f"\n\n  A) {session.options.get('A', '')}\n"
            f"  B) {session.options.get('B', '')}\n"
            f"  C) {session.options.get('C', '')}\n"
            f"  D) {session.options.get('D', '')}"
        )
        return response.message + choices

    # ── EXPLAIN (forced after max attempts) ───────────────────────────────────

    def _force_explain(self, session: DialogueSession, user_input: str) -> str:
        """
        Called when attempts >= 4.
        ScaffoldingEngine will choose strategy=EXPLAIN (attempts >= 4),
        so the LLM is allowed to reveal the answer directly.
        An LLM-generated reflection prompt follows instead of a hardcoded string.
        Transitions to REFLECTION afterward.
        """
        response = self.scaffold.generate_support(session, user_input)
        session.scaffold_level    = response.context.scaffold_level
        session.scaffold_strategy = response.strategy
        session.concepts_used     = [n.label for n in response.context.concept_chain]
        session.mode              = "REFLECTION"

        # ── LLM: personalised reflection prompt after the forced explanation ──
        closing = chat(
            session.session_id,
            (
                f"Topic: {session.topic}\n"
                f"Question: {session.current_question}\n"
                f"Correct answer: {session.correct_option}) {session.correct_answer}\n"
                f"Explanation just given: {response.message}\n\n"
                "Write a short reflection prompt for the student."
            ),
            system_prompt=_sys("POST_EXPLAIN"),
        )

        return (
            f"{response.message}\n\n"
            f"The correct answer was {session.correct_option}) {session.correct_answer}.\n\n"
            f"{closing}"
        )

    def _handle_explain(self, session: DialogueSession, user_input: str) -> str:
        """
        Handles the rare case where mode lands on EXPLAIN via the dispatcher
        (e.g. a future extension explicitly sets mode='EXPLAIN').
        Delegates to _force_explain so behaviour is consistent.
        """
        return self._force_explain(session, user_input)

    # ── REFLECTION ────────────────────────────────────────────────────────────

    def _handle_reflection(self, session: DialogueSession, user_input: str) -> str:
        """
        Short consolidation phase.
        - "next question" (and synonyms) → load a new question
        - Anything else                  → one LLM follow-up, stay in REFLECTION
        """
        if _is_next_q(user_input, session.session_id):
            return self._load_new_question(session)

        prompt = (
            f"Topic: {session.topic}\n"
            f"Question discussed: {session.current_question}\n"
            f"Correct answer: {session.correct_option}) {session.correct_answer}\n"
            f"Student's reflection: {user_input}\n\n"
            "Ask one short follow-up reflection question, "
            "or invite them to type 'next question' to continue practising."
        )
        return chat(session.session_id, prompt, system_prompt=_sys("REFLECTION"))