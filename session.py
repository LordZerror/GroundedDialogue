import uuid
from dataclasses import dataclass, field

@dataclass
class DialogueSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mode: str = "ONBOARDING"   # ONBOARDING, ASSESSMENT, SCAFFOLD, EXPLAIN, REFLECTION
    last_mode: str | None = None

    topic: str | None = None
    proficiency: str | None = None

    current_question: str | None = None
    correct_answer: str | None = None

    attempts: int = 0
    scaffold_level: int = 0

    history: list = field(default_factory=list)
    concepts_used: list = field(default_factory=list)