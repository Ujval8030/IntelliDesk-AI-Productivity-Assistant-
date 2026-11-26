import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


@app.route("/")
def home():
    return render_template("index.html")


@app.post("/api/assistant")
def assistant():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "chat")
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text first."}), 400

    if not client:
        return jsonify({
            "error": "OpenAI API key is not configured. Add OPENAI_API_KEY to your .env file."
        }), 500

    prompts = {
        "chat": (
            "You are IntelliDesk, a helpful AI productivity assistant. "
            "Give clear, practical and concise answers to the user's request."
        ),
        "summarize": (
            "Summarize the user's text in simple, clear language. "
            "Use short bullet points and preserve the important information."
        ),
        "rewrite": (
            "Rewrite the user's text to make it clear, polished and professional. "
            "Keep the original meaning. Return only the rewritten text."
        ),
        "study": (
            "Act as a friendly study assistant. Explain the topic in beginner-friendly "
            "language with a short explanation, key points and one simple example when useful."
        )
    }

    if action not in prompts:
        return jsonify({"error": "Unsupported assistant action."}), 400

    try:
        response = client.responses.create(
            model=MODEL,
            instructions=prompts[action],
            input=text
        )
        return jsonify({"result": response.output_text})
    except Exception as error:
        app.logger.exception("OpenAI request failed")
        return jsonify({
            "error": "The AI request could not be completed. Check your API key, model access, and connection."
        }), 502


if __name__ == "__main__":
    app.run(debug=True)
