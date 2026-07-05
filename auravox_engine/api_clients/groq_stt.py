import os
import io
import wave
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Module-level client avoids re-initialization on every chunk
_api_key = os.getenv("GROQ_API_KEY")
_client = Groq(api_key=_api_key) if _api_key else None


def transcribe_audio_chunk(audio_bytes: bytes) -> str:
    """Transcribe raw Tamil audio bytes using Groq Whisper.

    Uses language lock, a conditioning prompt, and zero temperature to
    aggressively suppress hallucinations (Arabic leakage, phonetic drift).
    """
    if not _client:
        raise ValueError("GROQ_API_KEY environment variable not set")

    # Package raw PCM into a valid in-memory WAV file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit PCM
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
    wav_buffer.seek(0)

    # Conditioning prompt: anchors Whisper's decoder to correct Tamil
    # phonetics and spelling, drastically reducing hallucinated output.
    _WHISPER_PROMPT = (
        "இது ஒரு தெளிவான தமிழ் உரையாடல். "
        "வணக்கம், எப்படி இருக்கிறீர்கள், நான் நன்றாக இருக்கிறேன். "
        "Tamil speech without background noise."
    )

    transcription = _client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=("chunk.wav", wav_buffer.read(), "audio/wav"),
        language="ta",
        prompt=_WHISPER_PROMPT,
        temperature=0.0,
        response_format="text",
    )
    return transcription.strip()