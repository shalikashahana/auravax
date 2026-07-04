import os
import requests

api_key = os.environ.get("CARTESIA_API_KEY")

def generate_emotional_audio(english_text: str, emotion: str) -> bytes:
    """Generates TTS using the pre-registered voice from CARTESIA_VOICE_ID via REST API."""
    try:
        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "X-API-Key": api_key,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json"
        }
        payload = {
            "model_id": "sonic-3.5",
            "transcript": english_text,
            "voice": {"mode": "id", "id": os.environ.get("CARTESIA_VOICE_ID")},
            "language": os.environ.get("TARGET_LANG", "en").lower()[:2],
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"Cartesia REST API Error: {response.status_code} - {response.text}")
            return b""
            
        return response.content
        
    except Exception as e:
        print(f"Cartesia Generation Error: {e}")
        return b""