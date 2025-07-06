import os
from flask import Flask, request, jsonify, abort
from dotenv import load_dotenv
from functools import wraps
import google.generativeai as genai

# --- Load environment variables ---
load_dotenv()

app = Flask(__name__)

# --- Configurations ---
API_KEY = os.getenv("GEMINI_API_KEY")
API_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET")

if not API_KEY:
    raise ValueError("AI_API_KEY not found in .env file.")
if not API_SECRET:
    raise ValueError("API_ACCESS_SECRET not found in .env file.")

# --- Initialize AI Model ---
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# --- Security Middleware ---
def require_api_secret(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.headers.get("X-API-Access-Secret") != API_SECRET:
            abort(403, description="Unauthorized access.")
        return f(*args, **kwargs)
    return wrapper

# --- Routes ---
@app.route('/')
def index():
    return jsonify({
        "status": "ok",
        "message": "Welcome to the AI Text API. Use /process to submit your text task."
    })

@app.route('/process', methods=['POST'])
@require_api_secret
def process_text():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request must be JSON."}), 400

    data = request.get_json()
    input_text = data.get('text')
    target_language = data.get('target_language')
    source_language = data.get('source_language', 'auto-detect')

    if not input_text or not target_language:
        return jsonify({"status": "error", "message": "Missing required fields: 'text' and 'target_language'."}), 400

    prompt = (
        f"Translate this from {source_language} to {target_language}. "
        "Only return the translated version without extra comments or punctuation. "
        f"Input: '{input_text}'"
    )

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()

        if not result or "I am not able to" in result:
            return jsonify({"status": "error", "message": "Processing failed."}), 500

        return jsonify({
            "status": "success",
            "input": input_text,
            "output": result,
            "language": target_language
        })

    except Exception as e:
        app.logger.error(f"AI Model Error: {e}")
        return jsonify({"status": "error", "message": f"Internal error: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
