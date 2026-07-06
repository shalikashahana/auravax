import os
import time
import concurrent.futures
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

# ---------------------------------------------------------------------------
# Circuit Breaker — Enterprise-grade Gemini overload protection
# ---------------------------------------------------------------------------
# States:
#   CLOSED  (normal)    → CIRCUIT_BREAKER_ACTIVE = False, calls proceed
#   OPEN    (tripped)   → CIRCUIT_BREAKER_ACTIVE = True,  bypass instantly
#   HALF-OPEN (probing) → 60s cooldown expired, try one call to see if
#                          Gemini is back; re-trip on failure, close on success
# ---------------------------------------------------------------------------
CIRCUIT_BREAKER_ACTIVE = False
CIRCUIT_BREAKER_TIME = 0
_COOLDOWN_SECONDS = 60

# ---------------------------------------------------------------------------
# Aggressive Timeout — 1-second hard wall-clock deadline per key attempt
# ---------------------------------------------------------------------------
_GEMINI_TIMEOUT_S = 1.0
_timeout_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="GeminiTimeout"
)


def _call_gemini(key: str, raw_transcript: str) -> str:
    """Execute a single Gemini API call (runs inside the timeout executor)."""
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


def gemini_grammar_filter(raw_transcript: str) -> str:
    """Clean raw Tamil STT text using Gemini as a grammar/slang filter.

    Circuit Breaker flow:
      1. If breaker is OPEN and within 60s cooldown → return raw_transcript
         instantly (0ms latency, zero API calls).
      2. If 60s have elapsed → enter HALF-OPEN state and try one call.
      3. On success → close the breaker (reset).
      4. On failure of ALL keys → trip the breaker (set OPEN + timestamp).

    Each key attempt is capped at 1.0s via ThreadPoolExecutor future timeout.
    """
    global CIRCUIT_BREAKER_ACTIVE, CIRCUIT_BREAKER_TIME

    if not AVAILABLE_KEYS:
        return raw_transcript

    # ── Circuit Breaker gate ──────────────────────────────────────────────
    if CIRCUIT_BREAKER_ACTIVE:
        elapsed = time.time() - CIRCUIT_BREAKER_TIME
        if elapsed < _COOLDOWN_SECONDS:
            # OPEN state — instant bypass, 0ms added latency
            remaining = int(_COOLDOWN_SECONDS - elapsed)
            print(
                f"🔴 Circuit breaker OPEN — bypassing Gemini "
                f"({remaining}s until retry)"
            )
            return raw_transcript
        else:
            # Cooldown expired → HALF-OPEN: reset and probe
            print("🟡 Circuit breaker HALF-OPEN — probing Gemini...")
            CIRCUIT_BREAKER_ACTIVE = False

    # ── Try each key with aggressive 1.0s timeout ────────────────────────
    for idx, key in enumerate(AVAILABLE_KEYS, start=1):
        try:
            future = _timeout_executor.submit(_call_gemini, key, raw_transcript)
            # ── THE GUILLOTINE: 1-second hard wall-clock deadline ──
            result = future.result(timeout=_GEMINI_TIMEOUT_S)

            # Success — ensure breaker is fully closed
            if CIRCUIT_BREAKER_ACTIVE:
                print("🟢 Circuit breaker CLOSED — Gemini recovered.")
                CIRCUIT_BREAKER_ACTIVE = False
            return result

        except concurrent.futures.TimeoutError:
            future.cancel()
            next_idx = idx + 1
            if next_idx <= len(AVAILABLE_KEYS):
                print(
                    f"⏱️  Gemini Key {idx} TIMED OUT (>{_GEMINI_TIMEOUT_S}s), "
                    f"switching to Key {next_idx}..."
                )
            else:
                print(
                    f"⏱️  Gemini Key {idx} TIMED OUT (>{_GEMINI_TIMEOUT_S}s). "
                    f"All keys exhausted — tripping circuit breaker."
                )
            continue

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
                    f"All keys exhausted — tripping circuit breaker."
                )
            continue

    # ── All keys failed — TRIP the circuit breaker ────────────────────────
    CIRCUIT_BREAKER_ACTIVE = True
    CIRCUIT_BREAKER_TIME = time.time()
    print(
        f"🔴 Circuit breaker TRIPPED — Gemini bypassed for "
        f"{_COOLDOWN_SECONDS}s cooldown."
    )
    return raw_transcript