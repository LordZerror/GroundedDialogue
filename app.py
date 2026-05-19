from flask import Flask, jsonify, render_template, request

from tutoringSystem import MisconceptionTutoringSystem

app = Flask(__name__)
tutor = MisconceptionTutoringSystem()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/start", methods=["POST"])
def start():
    session_id, greeting = tutor.start_conversation()
    return jsonify({"session_id": session_id, "reply": greeting})


@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    user_input = data.get("message", "")
    if not session_id or session_id not in tutor.sessions:
        return jsonify({"error": "invalid session"}), 400
    reply = tutor.process_user_input(session_id, user_input)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
