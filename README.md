# GroundedDialogue: Misconception Tutoring System

An AI-powered Intelligent Tutoring System (ITS) designed to provide empathetic, adaptive, and conceptually grounded Computer Science education. GroundedDialogue leverages the Computer Science Ontology (CSO), local Large Language Models (LLMs), and Bloom's Taxonomy to dynamically generate curriculum and scaffold student learning through interactive, mixed-initiative dialogue.

GroundedDialogue is created to apply pedagogical techniques combined with state-of-the-art Generative AI. It showcases structured outputs, rigorous evaluation, and complex dialogue state management.

## 🚀 Key Features

- **Dynamic MCQ Generation:** Generates Multiple Choice Questions for any CS topic on-the-fly, adapting to the student's proficiency level (Beginner, Intermediate, Proficient) using Bloom's Taxonomy.
- **Ontology-Grounded Scaffolding:** Uses the Computer Science Ontology (CSO) to ground hints and explanations, ensuring pedagogical accuracy.
- **Adaptive Scaffolding Strategies:** Gradually steps up the level of support based on student performance (Socratic -> Contrastive -> Worked Example -> Direct Explanation).
- **Mixed-Initiative Dialogue:** Students can pause the current assessment at any time to ask conceptual clarification questions, and the system seamlessly returns to the problem once answered.
- **Empathetic Persona:** The AI provides warm, conversational encouragement and invites reflective thinking without simply handing over the answers.
- **Privacy-First & Local:** Uses a local `llama.cpp` server to run the core models, removing external API dependencies and protecting user data.
- **Robust Evaluation:** Evaluated using G-Eval frameworks to ensure accurate mode compliance, format constraints, and no-leak policies during tutoring.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **LLM Engine:** Local `llama.cpp` server
- **Knowledge Graph:** RDFLib with the Computer Science Ontology (CSO)
- **Evaluation:** DeepEval (for rubric-based G-Eval), scikit-learn, pandas
- **Frontend:** HTML/CSS & CLI interface

## ⚙️ Setup and Installation

### Prerequisites
1. Python 3.9+
2. `llama.cpp` installed and accessible.
3. The Computer Science Ontology file (`CSO 3.ttl`) should be placed in the project root.

### 1. Install Dependencies
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Start the Local LLM Server
Start a local `llama.cpp` server on port `8080`.
Example command (assuming you have a compatible GGUF model):
```bash
./llama-server \       
-hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
--ctx-size 16384 \
--top-p 0.8 \
--top-k 20 \
--min-p 0.00 \
--chat-template-kwargs '{"enable_thinking": false}'


```
*Note: The tutor relies on this server running locally at `http://127.0.0.1:8080` for generating questions and handling dialogue logic.*

## 💻 Usage

You can run the tutor in either Command Line Interface (CLI) mode or via the Web Application.

### CLI Mode
Run the interactive CLI loop in your terminal:
```bash
python main.py
```
*Follow the prompt to set your topic and proficiency level, and begin answering questions. Type `quit` or `exit` to stop.*

### Web Application
Start the Flask web server:
```bash
python app.py
```
Then, open your browser and navigate to `http://localhost:5000` to interact with the tutor via the web UI.

## 🧪 Evaluation Framework

This repository includes a rigorous evaluation suite located in the `eval/` directory to measure the correctness and safety of the dialogue system:
- **Test Isolation:** Ensures dialogue management tests are separated from MCQ generation using pinned questions.
- **G-Eval Rubrics:** Tests for *Mode Appropriateness*, *No Answer Leaks*, and *Format Compliance* using frontier LLMs as judges.
- **Cohen's Kappa:** Validates LLM judge ratings against human calibration scores to ensure defensibility of results.

To run the evaluations (requires `OPENAI_API_KEY` for the judge model):
1. Harvest transcripts: `python -m eval.harvest`
2. Run G-Eval scoring: `python -m eval.run_geval`
3. Compute Cohen's Kappa: `python -m eval.kappa`

---
*Created with ❤️ for students, educators, and the open-source community.*
