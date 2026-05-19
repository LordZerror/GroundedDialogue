from __future__ import annotations

import json
import random

import requests

DEFAULT_URL = 'http://127.0.0.1:8080'
DEFAULT_HEADERS = { 'Content-Type': 'application/json' }
_chat_histories = {}
DEFAULT_SYSTEM_PROMPT = "You are a CS tutoring assistant."

# Part 1: Simple no-context query

def ask(prompt):
    message = complete_messages([
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": prompt },
    ])
    return message["content"].strip()

# print(ask("What is the capital of Oklahoma?"))

def complete_messages(messages, response_format=None):
    data = {"messages": messages}
    if response_format is not None:
        data["response_format"] = response_format

    response = requests.post(
        f"{DEFAULT_URL}/v1/chat/completions",
        headers=DEFAULT_HEADERS,
        json=data,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


# Part 2: Multi-turn chats
def chat(session_id: str, prompt: str, system_prompt: str | None = None, reset: bool = False) -> str:
    global _chat_histories

    if reset or session_id not in _chat_histories:
        _chat_histories[session_id] = [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}
        ]
    elif system_prompt and _chat_histories[session_id][0]["content"] != system_prompt:
        _chat_histories[session_id][0] = {"role": "system", "content": system_prompt}

    _chat_histories[session_id].append({"role": "user", "content": prompt})

    message = complete_messages(_chat_histories[session_id])
    _chat_histories[session_id].append(message)
    return message["content"].strip()


Question_Answer = {
    "type": "object",
    "properties": {
        "skillLevel": {
            "type": "string",
            "enum": ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        },
        "question": {
            "type": "string"
        },
        "options": {
            "type": "object",
            "properties": {
                "A": {"type": "string"},
                "B": {"type": "string"},
                "C": {"type": "string"},
                "D": {"type": "string"}
            },
            "required": ["A", "B", "C", "D"],
            "additionalProperties": False
        },
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D"]
        }
    },
    "required": ["skillLevel", "question", "options", "answer"],
    "additionalProperties": False
}

Remember_question = "What is the definition of 'overfitting' in machine learning? Choose the correct option."
Understand_question = "What does 'overfitting' mean in the context of training a machine learning model? Choose the best definition." 
Apply_question = "Which scenario best illustrates 'overfitting' in a machine learning model?"
Analyze_question = "What can you infer about the impact of 'overfitting' on a model's performance?"
Evaluate_question = "What criteria would you use to assess the effectiveness of a model in handling overfitting, and what changes would you recommend to improve its performance?"
Create_question = "How would you design a plan to prevent overfitting in a machine learning model while maintaining its predictive accuracy?"


bloom_prompt =  f"""
The following are Bloom’s skills and some example questions showing the framework fitting each skill level:

1. Skill: Remember, Example: {Remember_question}
2. Skill: Understand, Example: {Understand_question}
3. Skill: Apply, Example: {Apply_question}
4. Skill: Analyze, Example: {Analyze_question}
5. Skill: Evaluate, Example: {Evaluate_question}
6. Skill: Create, Example: {Create_question}
These questions are created to evaluate students on a range of cognitive skills, from basic knowledge to critical thinking and problem-solving.
You are supposed to create ONE multiple choice question (with options) corresponding to the user_skill level in revised Bloom’s taxonomy.

CRITICAL INSTRUCTIONS FOR ORIGINALITY:
1. The examples above are strictly to illustrate the *cognitive depth* required for each level. 
3. Write an original, highly specific, and creative question about the User_Topic. 
4. The question MUST be exclusively about the User_Topic. 
Keep the answer choices short and to the point.
User_Topic = {{topic}}
user_skill = {{skill}}
"""

#dictionary of dictionaries - to increase difficulty level as per user request
profSkillMapping = {
    "beginner": ["Remember", "Understand"],
    "intermediate": ["Remember", "Understand", "Apply", "Analyze"],
    "proficient": ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
}

def retrieveQuestion(topic, profLevel):
    skillToTest=random.choice(profSkillMapping[profLevel])
    updated_prompt = bloom_prompt.replace("{topic}", topic).replace("{skill}", skillToTest)
    return bloomGeneration(topic, prompt=updated_prompt, schema=Question_Answer)

def bloomGeneration(topic, prompt, schema):
    updated_prompt = prompt.replace("{topic}", topic)
    message = complete_messages(
        [
            {
                "role": "system",
                "content": "You are an experienced computer science instructor for a graduate-level course based in the Computer Science Ontology by Knowledge Media Institute",
            },
            {"role": "user", "content": updated_prompt},
        ],
        response_format={"type": "json_object", "schema": schema},
    )
    return message["content"].strip()
