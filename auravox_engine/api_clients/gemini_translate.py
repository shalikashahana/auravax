import os
import re
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Safe English fallback — guarantees Cartesia never receives unsupported text
_SAFE_FALLBACK = {"translation": "I didn't quite catch that.", "emotion": "neutral"}

# Module-level client (created once per process)
_api_key = os.getenv("GEMINI_API_KEY")
_client = genai.Client(api_key=_api_key) if _api_key else None


def _build_system_instruction() -> str:
    """Build a dynamic system prompt based on configured language pair."""
    source_lang = os.environ.get("SOURCE_LANG", "ta")
    target_lang = os.environ.get("TARGET_LANG", "en")
    return (
        f"You are a strict real-time translator. "
        f"Translate the input text from language code '{source_lang}' to language code '{target_lang}'. "
        f"Also detect the emotional tone of the original sentence "
        f"(anger, sadness, happiness, surprise, neutral). "
        f"Return ONLY a raw JSON dictionary without markdown wrappers: "
        f'{{"translation": "...", "emotion": "..."}} and nothing else.'
    )


def translate_and_extract_emotion(text: str) -> dict:
    """Translate source text and extract emotion using Gemini.
    Returns a dictionary with keys 'translation' and 'emotion'.
    On any failure, returns a safe fallback to protect downstream TTS.
    """
    if not _client:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    try:
        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=_build_system_instruction(),
                response_mime_type="application/json",
            ),
        )
        # response_mime_type="application/json" guarantees valid JSON from Gemini,
        # but we still extract defensively in case of any wrapper noise
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in Gemini response")
        result = json.loads(match.group(0))
        if not isinstance(result, dict) or "translation" not in result or "emotion" not in result:
            raise ValueError("Invalid JSON structure")
        return result
    except Exception as e:
        print(f"Gemini Translation Error: {e}")
        return dict(_SAFE_FALLBACK)