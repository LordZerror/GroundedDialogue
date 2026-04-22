import requests

DEFAULT_URL = 'http://127.0.0.1:8080'
DEFAULT_HEADERS = { 'Content-Type': 'application/json' }

# Part 1: Simple no-context query

def ask(prompt):
    data = { "messages": [ 
                { "role": "system", "content": "You are a helpful assistant." }, # First message is always the system prompt
                { "role": "user", "content": prompt }
                ]
            }
    response = requests.post(f"{DEFAULT_URL}/v1/chat/completions", 
                             headers=DEFAULT_HEADERS, 
                             json=data)
    response.raise_for_status()
    text = response.json()['choices'][0]['message']['content'].strip()
    return(text)

# print(ask("What is the capital of Oklahoma?"))

# Part 2: Multi-turn chats

_chat_histories = dict()

def chat(id, prompt):
    global _chat_histories
    if id not in _chat_histories: 
        _chat_histories[id] = [ { "role": "system", 
                                 "content": "You are a helpful assistant." } ]
    user_message = { "role": "user", "content": prompt }
    _chat_histories[id].append(user_message)
    data = { "messages": _chat_histories[id] }
    response = requests.post(f"{DEFAULT_URL}/v1/chat/completions", 
                             headers=DEFAULT_HEADERS, 
                             json=data)
    response.raise_for_status
    _chat_histories[id].append(response.json()['choices'][0]['message'])
    text = response.json()['choices'][0]['message']['content'].strip()
    return(text)

#print(chat("oklahoma", "What is the capital of Oklahoma?"))
#print(chat("oklahoma", "What state is just south of there?"))

# Part 3: JSON Schema-constrained generation

import json

def ask_schema(prompt, schema):
    data = { "messages": [ 
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": prompt }
                ],
             "response_format": {"type": "json_object", "schema": schema}
            }
    response = requests.post(f"{DEFAULT_URL}/v1/chat/completions", 
                             headers=DEFAULT_HEADERS, 
                             json=data)
    response.raise_for_status()
    text = response.json()['choices'][0]['message']['content'].strip()
    obj = json.loads(text)
    return(obj)

YES_NO_SCHEMA = { "type": "string", "enum": ["yes", "no"] }


def ask_schema_pronouns(prompt, pronounEnum):
    pronounSchema = { "type": "string", "enum": pronounEnum }

    data = { "messages": [ 
                { "role": "system", "content": "You are an expert in pronoun resolution." },
                { "role": "user", "content": prompt }
                ],
             "response_format": {"type": "json_object", "schema": pronounSchema}
            }
    response = requests.post(f"{DEFAULT_URL}/v1/chat/completions", 
                             headers=DEFAULT_HEADERS, 
                             json=data)
    response.raise_for_status()
    text = response.json()['choices'][0]['message']['content'].strip()
    obj = json.loads(text)
    return(obj)

Question_Answer = {
    "type": "object",
    "properties": {
        "Questions": {
            "type": "array",
            "items": {
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
                    }
                },
                "required": ["skillLevel", "question", "options"],
                "additionalProperties": False
            }
        }
    },
    "required": ["Questions"],
    "additionalProperties": False
}

Remember_question = "What is the definition of 'overfitting' in machine learning? Choose the correct option."
Understand_question = "What does 'overfitting' mean in the context of training a machine learning model? Choose the best definition." 
Apply_question = "Which scenario best illustrates 'overfitting' in a machine learning model?"
Analyze_question = "What can you infer about the impact of 'overfitting' on a model's performance?"
Evaluate_question = "What criteria would you use to assess the effectiveness of a model in handling overfitting, and what changes would you recommend to improve its performance?"
Create_question = "How would you design a plan to prevent overfitting in a machine learning model while maintaining its predictive accuracy?"


bloom_prompt =  f"""
The following are Bloom’s skills and some example questions corresponding to different levels if the User_Topic was 'overfitting':

1. Skill: Remember, Example: {Remember_question}
2. Skill: Understand, Example: {Understand_question}
3. Skill: Apply, Example: {Apply_question}
4. Skill: Analyze, Example: {Analyze_question}
5. Skill: Evaluate, Example: {Evaluate_question}
6. Skill: Create, Example: {Create_question}
User_Topic = {{topic}}
You are supposed to create one multiple choice question along with options corresponding to each level in revised Bloom’s taxonomy for the User_Topic. 
Keep the answer choices short and to the point.
These questions are created to evaluate students on a range of cognitive skills, from basic knowledge to critical thinking and problem-solving.
"""


def bloomGeneration(topic, prompt, schema):
    updated_prompt = prompt.replace("{topic}", topic)
    data = { "messages": [ 
                { "role": "system", "content": "You are an experienced computer science instructor for a graduate-level course based in the Computer Science Ontology by Knowledge Media Institute" }, # First message is always the system prompt
                { "role": "user", "content": updated_prompt }
                ],
                "response_format": {"type": "json_object", "schema": schema}

            }
    response = requests.post(f"{DEFAULT_URL}/v1/chat/completions", 
                             headers=DEFAULT_HEADERS, 
                             json=data)
    response.raise_for_status()
    text = response.json()['choices'][0]['message']['content'].strip()
    return(text)

print(bloomGeneration("neural networks", bloom_prompt, Question_Answer))

print(bloomGeneration("knowledge graphs", bloom_prompt, Question_Answer))