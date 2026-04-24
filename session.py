from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class DialogueSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # States: ONBOARDING → ASSESSMENT → SCAFFOLD → REFLECTION
    #                                 ↘ EXPLAIN  ↗
    mode: str = "ONBOARDING"
    last_mode: str | None = None          # used for mixed-initiative return

    # Set during onboarding
    topic: str | None = None
    proficiency: str | None = None        # "Beginner" | "Intermediate" | "Proficient"

    # Set when a question is loaded
    question_concept: str | None = None
    bloom_level: str | None = None
    current_question: str | None = None
    options: dict = field(default_factory=dict)   # {"A": "...", "B": "...", ...}
    correct_option: str | None = None             # "A" | "B" | "C" | "D"
    correct_answer: str | None = None             # full text of correct option

    # Scaffold tracking
    attempts: int = 0
    scaffold_level: str | None = None
    scaffold_strategy: str | None = None
    concepts_used: list = field(default_factory=list)

    # Conversation history (unused by LLM helpers but useful for logging)
    history: list = field(default_factory=list)
