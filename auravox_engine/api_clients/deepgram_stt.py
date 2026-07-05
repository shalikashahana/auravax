import os
import io
import wave
import requests
from dotenv import load_dotenv

load_dotenv()

# Deepgram REST endpoint — 'enhanced' model with native Tamil support
_DEEPGRAM_URL = (
    "https://api.deepgram.com/v1/listen"
    "?model=enhanced"
    "&language=ta"
    "&punctuate=true"
)

_api_key = os.environ.get("DEEPGRAM_API_KEY")
if not _api_key:
    print("⚠️  DEEPGRAM_API_KEY not found. STT will not work.")


def transcribe_audio_chunk(audio_bytes: bytes) -> str:
    """Transcribe raw Tamil audio bytes using Deepgram Nova-2 via REST API.

    Converts raw PCM-16 mono 16 kHz bytes into an in-memory WAV, then
    POSTs it to the Deepgram /listen endpoint. Returns the transcript
    string, or an empty string on any failure so the pipeline drops the
    chunk gracefully.
    """
    if not _api_key:
        print("❌ DEEPGRAM_API_KEY not set — cannot transcribe.")
        return ""

    # Package raw PCM into a valid in-memory WAV file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit PCM
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
    wav_bytes = wav_buffer.getvalue()

    try:
        resp = requests.post(
            _DEEPGRAM_URL,
            headers={
                "Authorization": f"Token {_api_key}",
                "Content-Type": "audio/wav",
            },
            data=wav_bytes,
            timeout=10,
        )
        resp.raise_for_status()

        transcript = (
            resp.json()
            .get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
        return transcript.strip()

    except Exception as e:
        # Print response body for debugging 4xx/5xx errors
        try:
            print(f"❌ Deepgram STT Error: {e} | Response: {resp.text}")
        except Exception:
            print(f"❌ Deepgram STT Error: {e}")
        return ""
