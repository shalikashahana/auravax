import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Safe English fallback — guarantees Cartesia never receives unsupported text
_SAFE_FALLBACK = {"translation": "I didn't quite catch that.", "emotion": "neutral"}

# Module-level Groq client (created once per process)
_api_key = os.environ.get("GROQ_API_KEY")
_client = Groq(api_key=_api_key) if _api_key else None

if not _api_key:
    print("⚠️  GROQ_API_KEY not found. Translation will not work.")

# Anti-hallucination system prompt
_SYSTEM_PROMPT = (
    "You are a strict, emotionless Tamil-to-English translation machine. "
    "Your ONLY purpose is to output the English translation of the "
    "provided Tamil text.\n"
    "CRITICAL RULES:\n"
    "1. NEVER converse with the user.\n"
    "2. NEVER say 'I don't understand', 'Could you clarify', or add "
    "any filler words.\n"
    "3. NEVER output Tamil characters. The output MUST be 100% English.\n"
    "4. If the input is incomplete, misspelled, or slang, translate it "
    "as literally and naturally as possible without complaining.\n"
    "Output ONLY the English string."
)


def translate_and_extract_emotion(tamil_text: str) -> dict:
    """Translate Tamil text to English using Groq LLaMA-3.

    Returns a dictionary with keys 'translation' and 'emotion'.
    Emotion is hardcoded to 'neutral' to eliminate overhead.
    On any failure, returns a safe fallback to protect downstream TTS.
    """
    if not _client:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Add it to your .env file."
        )

    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": tamil_text},
            ],
            temperature=0.3,
            max_tokens=512,
        )

        translated = response.choices[0].message.content.strip()
        if not translated:
            raise ValueError("Empty response from Groq")

        return {"translation": translated, "emotion": "neutral"}

    except Exception as e:
        print(f"❌ Groq Translation Error: {e}")
        return dict(_SAFE_FALLBACK)
