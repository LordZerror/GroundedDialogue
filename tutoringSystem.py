import json
from session import DialogueSession
from llmHelpers import chat, retrieveQuestion
#from scaffold import ScaffoldingEngine

def get_system_prompt(mode: str) -> str:
    prompts = {
        "ONBOARDING": (
            "You are an experienced computer science instructor meant to quiz user on a topi from Computer Science Ontology. Ask for a topic and user's proficiency level."
        ),
        "MCQ_GENERATION": (
            "You are an experienced computer science instructor generating a multiple-choice question."
        ),
        "REFLECTION": (
            "You are a reflective tutor. Ask the student what they learned and what changed."
        ),
        "EXPLAIN": (
            "You are a clear tutor. Explain simply and directly."
        ),
    }
    return prompts.get(mode, "You are a helpful assistant.")

class MisconceptionTutoringSystem:
    def __init__(self):
        self.sessions = {}
        #self.ontology = ontology_service
        #self.scaffold = ScaffoldingEngine(ontology_service, chat)

    def start_conversation(self):
        session = DialogueSession()
        self.sessions[session.session_id] = session
        print(f"current state is {session.mode}")
        return session.session_id, (
            "Hi! I’m your CS tutor.\n"
            "What topic would you like to practice?\n"
            "Also tell me your proficiency level: beginner, intermediate, or proficient."
        )

    def process_user_input(self, session_id: str, user_input: str) -> str:
        session = self.sessions[session_id]

        if session.mode == "ONBOARDING":
            return self._handle_onboarding(session, user_input)

        if session.mode == "ASSESSMENT":
            return self._handle_assessment(session, user_input)

        if session.mode == "SCAFFOLD":
            return self._handle_scaffold(session, user_input)

        if session.mode == "EXPLAIN":
            return self._handle_explain(session, user_input)

        if session.mode == "REFLECTION":
            return self._handle_reflection(session, user_input)

        return "I am not sure how to continue."

    def _handle_onboarding(self, session, user_input: str) -> str:
        print(f"current state is {session.mode}")
        #Todo - make an llm call to extract topic and proficiency level 
        parts = [p.strip() for p in user_input.split(",")]
        session.topic = parts[0]
        session.proficiency = parts[1] if len(parts) > 1 else "beginner"

        mcq = json.loads(retrieveQuestion(session.topic, session.proficiency))
        session.current_question = mcq["question"]
        session.correct_answer = mcq["answer"]
        session.mode = "ASSESSMENT"

        return (
            f"{mcq['question']}\n"
            f"A) {mcq['options']['A']}\n"
            f"B) {mcq['options']['B']}\n"
            f"C) {mcq['options']['C']}\n"
            f"D) {mcq['options']['D']}\n"
        )

    def _handle_assessment(self, session, user_input: str) -> str:
        print(f"current state is {session.mode}")

        session.attempts += 1

        #todo
        if user_input.strip().upper() == session.correct_answer.strip().upper():
            session.mode = "REFLECTION"
            return "Correct. What made that answer feel right to you?"

        session.scaffold_level = min(session.scaffold_level + 1, 3)
        session.mode = "SCAFFOLD"

        return self.scaffold.generate_support(session, user_input)

    def _handle_scaffold(self, session, user_input: str) -> str:
        print(f"current state is {session.mode}")

        session.attempts += 1

        if session.attempts >= 4:
            session.mode = "EXPLAIN"
            return self._handle_explain(session, user_input)

        return self.scaffold.generate_support(session, user_input)

    def _handle_explain(self, session, user_input: str) -> str:
        print(f"current state is {session.mode}")

        context = self.ontology.get_scaffold_context(session.topic, session.scaffold_level)

        prompt = f"""
Question: {session.current_question}
Correct answer: {session.correct_answer}
Student said: {user_input}
Ontology context: {context}

Explain the answer clearly and briefly, then ask one reflection question.
"""
        session.mode = "REFLECTION"
        return chat(session.session_id, prompt, system_prompt=get_system_prompt("EXPLAIN"))

    def _handle_reflection(self, session, user_input: str) -> str:
        print(f"current state is {session.mode}")

        session.last_mode = session.mode
        session.mode = "ASSESSMENT"

        prompt = f"""
The topic was {session.topic}.
The student reflected: {user_input}

Ask one short follow-up reflection question, or invite them to try another question.
"""
        return chat(session.session_id, prompt, system_prompt=get_system_prompt("REFLECTION"))