import os
import google.generativeai as genai
from flask import Flask, request, jsonify, abort
from dotenv import load_dotenv
from functools import wraps

# --- Initialization ---
load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAPIDAPI_PROXY_SECRET = os.getenv("RAPIDAPI_PROXY_SECRET")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in the .env file.")
if not RAPIDAPI_PROXY_SECRET:
    raise ValueError("RAPIDAPI_PROXY_SECRET not found. Please set it in the .env file.")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest') # Use flash for speed and cost-efficiency

# --- Security Decorator for RapidAPI ---
def require_rapidapi_secret(f):
    """A decorator to verify the RapidAPI Proxy Secret."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # The header name on RapidAPI is 'X-RapidAPI-Proxy-Secret'
        if request.headers.get('X-RapidAPI-Proxy-Secret') != RAPIDAPI_PROXY_SECRET:
            abort(403, description="Forbidden: You are not authorized to access this resource.")
        return f(*args, **kwargs)
    return decorated_function

# --- API Endpoints ---
@app.route('/')
def home():
    """A simple welcome message to show the API is running."""
    return jsonify({
        "status": "ok",
        "message": "Welcome to the Gemini Translator API! Use the /translate endpoint to translate text."
    })

@app.route('/translate', methods=['POST'])
@require_rapidapi_secret
def translate_text():
    """
    Translates a given text to a target language using the Gemini API.
    Expects a JSON body with 'text' and 'target_language'.
    'source_language' is optional.
    """
    # 1. Input Validation
    if not request.is_json:
        return jsonify({"status": "error", "message": "Invalid request: Missing JSON body."}), 400

    data = request.get_json()
    text_to_translate = data.get('text')
    target_language = data.get('target_language')
    source_language = data.get('source_language', 'auto-detect') # Default to auto-detect

    if not text_to_translate or not target_language:
        return jsonify({"status": "error", "message": "Missing required fields: 'text' and 'target_language'."}), 400

    # 2. Construct the Prompt for Gemini
    prompt = (
        f"Translate the following text from {source_language} to {target_language}. "
        "Provide ONLY the translated text, without any additional explanations, "
        "introductory phrases, or quotation marks. "
        f"Text to translate: '{text_to_translate}'"
    )

    # 3. Call the Gemini API
    try:
        response = model.generate_content(prompt)
        translated_text = response.text.strip()
        
        # A simple check to see if Gemini returned a reasonable response
        if not translated_text or "I am not able to" in translated_text:
             return jsonify({"status": "error", "message": "Translation failed. The model could not process the request."}), 500

        return jsonify({
            "status": "success",
            "original_text": text_to_translate,
            "target_language": target_language,
            "translated_text": translated_text
        })

    except Exception as e:
        # Catch potential errors from the Gemini API (e.g., rate limits, content filtering)
        app.logger.error(f"Gemini API error: {e}")
        return jsonify({"status": "error", "message": f"An error occurred during translation: {str(e)}"}), 500

# --- Health Check Endpoint (Good for deployment platforms) ---
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
