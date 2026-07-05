import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------------------------------------------------------------------------
# Dual-key failover pool for instant 429/503 bypass
# ---------------------------------------------------------------------------
_KEY_1 = os.getenv("GEMINI_API_KEY_1")
_KEY_2 = os.getenv("GEMINI_API_KEY_2")
AVAILABLE_KEYS = [k for k in [_KEY_1, _KEY_2] if k]

if not AVAILABLE_KEYS:
    print("⚠️  No Gemini API keys found. Grammar filter will pass through raw text.")

# System prompt: grammar filter only — NO translation
_SYSTEM_PROMPT = (
    "You are a grammar filter. The input is raw Tamil STT output. "
    "Fix phonetic errors and STT hallucinations. "
    "Do NOT translate. Output ONLY the clean Tamil text."
)


def gemini_grammar_filter(raw_transcript: str) -> str:
    """Clean raw Tamil STT text using Gemini as a grammar/slang filter.

    Tries each available API key in sequence. If a key fails (429, 503,
    or any error), instantly switches to the next key with ZERO delay.

    CRITICAL BYPASS: If ALL keys fail, returns the raw transcript as-is
    so the pipeline degrades gracefully without latency or error messages.
    """
    if not AVAILABLE_KEYS:
        # No keys configured — pass through raw text silently
        return raw_transcript

    for idx, key in enumerate(AVAILABLE_KEYS, start=1):
        try:
            client = genai.Client(api_key=key)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=raw_transcript,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="text/plain",
                ),
            )

            cleaned = response.text.strip()
            if not cleaned:
                raise ValueError("Empty response from Gemini")

            return cleaned

        except Exception as e:
            next_idx = idx + 1
            if next_idx <= len(AVAILABLE_KEYS):
                print(
                    f"⚡ Gemini Key {idx} failed ({type(e).__name__}), "
                    f"switching to Key {next_idx}..."
                )
            else:
                print(
                    f"⚡ Gemini Key {idx} failed ({type(e).__name__}). "
                    f"All keys exhausted — bypassing filter."
                )
            # NO sleep, NO delay — instant continue
            continue

    # All keys failed — graceful degradation: return raw text as-is
    return raw_transcript