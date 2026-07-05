"""The main orchestrator script"""
"""
AuraVox - Zero-Latency Real-Time Voice Translation
====================================================
Terminal (CLI) Orchestrator - main.py

Pipeline:
    Mic -> VAD Chunking -> [ThreadPoolExecutor Worker] -> Groq STT
         -> Gemini Translate+Emotion -> Cartesia Emotional TTS -> Speaker

Design notes:
    - The VAD capture loop runs on the MAIN thread and NEVER blocks on network I/O.
    - Each finalized speech chunk is handed off to a ThreadPoolExecutor so multiple
      chunks can be in-flight through the API pipeline simultaneously.
    - Playback is serialized through a dedicated single-consumer Queue + thread so
      translated audio doesn't overlap/garble on the speaker, even if two worker
      threads finish out of order.
    - Every external call (mic, STT, translate, TTS, playback) is wrapped so a
      single failure degrades gracefully instead of killing the whole process.
"""

import sys
# Fix Windows console encoding — cp1252 cannot render emoji characters
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
import queue
import signal
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

try:
    import pyaudio
except ImportError:
    print("❌ Missing dependency 'pyaudio'. Install with: pip install pyaudio")
    sys.exit(1)

try:
    import webrtcvad
except ImportError:
    print("❌ Missing dependency 'webrtcvad'. Install with: pip install webrtcvad")
    sys.exit(1)

# --- AuraVox API client modules (assumed built & importable) ---------------
try:
    from api_clients import deepgram_stt
    from api_clients import gemini_translate
    from api_clients import groq_translate
    from api_clients import cartesia_tts
except ImportError as e:
    print(f"❌ Could not import api_clients package: {e}")
    print("   Make sure main.py sits alongside the 'api_clients' folder.")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    RATE = 16000                 # webrtcvad requires 8000/16000/32000/48000 Hz
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    SAMPLE_WIDTH = 2              # bytes per sample for paInt16

    FRAME_DURATION_MS = 30        # webrtcvad supports 10/20/30 ms frames
    FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)          # samples/frame
    FRAME_BYTES = FRAME_SIZE * SAMPLE_WIDTH                     # bytes/frame

    VAD_AGGRESSIVENESS = 3        # 0 (lenient) - 3 (most aggressive filtering)
    SILENCE_TIMEOUT_MS = 1000     # 1s — matches natural sentence pause (0.8-1.2s)
    SILENCE_FRAMES = max(1, int(SILENCE_TIMEOUT_MS / FRAME_DURATION_MS))

    MIN_CHUNK_MS = 250            # ignore chunks shorter than this (noise/blips)
    MIN_CHUNK_FRAMES = max(1, int(MIN_CHUNK_MS / FRAME_DURATION_MS))

    MAX_WORKERS = 4               # concurrent in-flight pipeline chunks

    INPUT_DEVICE_INDEX = None     # None = system default mic
    OUTPUT_DEVICE_INDEX = None    # None = system default speaker


# ============================================================================
# TERMINAL UX HELPERS
# ============================================================================
_print_lock = threading.Lock()


def log(msg: str) -> None:
    """Thread-safe, timestamped console output."""
    with _print_lock:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


# ============================================================================
# AUDIO ENGINE (mic capture + speaker playback helpers)
# ============================================================================
class AudioEngine:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self._detect_virtual_audio_cable()

    def _detect_virtual_audio_cable(self) -> None:
        """Scan audio devices for a Virtual Audio Cable output.

        If a device whose name contains 'CABLE Output' or 'Virtual' is found,
        Config.OUTPUT_DEVICE_INDEX is set to its index so that TTS audio is
        routed through the virtual cable (Google Meet picks it up as a mic).
        Falls back to the system default speaker if no match is found.
        """
        device_count = self.pa.get_device_count()
        for i in range(device_count):
            try:
                info = self.pa.get_device_info_by_index(i)
                name = info.get("name", "").lower()
                max_output = info.get("maxOutputChannels", 0)

                # Only consider devices that actually support output
                if max_output < 1:
                    continue

                if "cable output" in name or "virtual" in name:
                    Config.OUTPUT_DEVICE_INDEX = i
                    log(f"🔌 Virtual Audio Cable detected: "
                        f"\"{info['name']}\" (device index {i})")
                    return
            except Exception:
                # Corrupt / unavailable device entry — skip silently
                continue

        # No virtual cable found — fall back to system default
        Config.OUTPUT_DEVICE_INDEX = None
        log("⚠️  No Virtual Audio Cable found (looked for 'CABLE Output' / "
            "'Virtual'). Falling back to default speaker output. "
            "Install VB-Cable for Google Meet routing.")

    def open_input_stream(self):
        return self.pa.open(
            format=Config.FORMAT,
            channels=Config.CHANNELS,
            rate=Config.RATE,
            input=True,
            input_device_index=Config.INPUT_DEVICE_INDEX,
            frames_per_buffer=Config.FRAME_SIZE,
        )

    def open_output_stream(self):
        return self.pa.open(
            format=Config.FORMAT,
            channels=Config.CHANNELS,
            rate=Config.RATE,
            output=True,
            output_device_index=Config.OUTPUT_DEVICE_INDEX,
        )

    def play(self, audio_bytes: bytes) -> None:
        """Blocking playback of raw PCM16 mono 16kHz bytes."""
        if not audio_bytes:
            return
        stream = self.open_output_stream()
        try:
            chunk_size = 4096
            for i in range(0, len(audio_bytes), chunk_size):
                stream.write(audio_bytes[i:i + chunk_size])
        finally:
            stream.stop_stream()
            stream.close()

    def close(self):
        self.pa.terminate()


# ============================================================================
# PIPELINE WORKER (runs inside ThreadPoolExecutor)
# ============================================================================
def process_chunk(chunk_id: int, audio_bytes: bytes,
                   playback_queue: "queue.Queue") -> None:
    """
    Full pipeline for a single VAD-cut speech chunk. Designed to fail soft:
    any stage failing simply drops this chunk and logs a warning, without
    affecting other in-flight chunks or the capture loop.
    """
    try:
        log(f"🚀 [chunk {chunk_id}] Sending to Deepgram STT...")
        tamil_text = deepgram_stt.transcribe_audio_chunk(audio_bytes)

        if not tamil_text or not tamil_text.strip():
            log(f"🤔 [chunk {chunk_id}] No speech detected by STT — skipping.")
            return
        log(f"📝 [chunk {chunk_id}] Transcribed: \"{tamil_text.strip()}\"")

    except Exception as e:
        log(f"❌ [chunk {chunk_id}] Deepgram STT failed: {e}")
        return

    # --- Stage 2: Gemini Grammar Filter (optional pre-filter) ---------------
    try:
        log(f"🧹 [chunk {chunk_id}] Gemini grammar filter...")
        cleaned_tamil = gemini_translate.gemini_grammar_filter(tamil_text)
        log(f"✨ [chunk {chunk_id}] Cleaned: \"{cleaned_tamil.strip()}\"")
    except Exception as e:
        log(f"⚠️  [chunk {chunk_id}] Grammar filter error — using raw text: {e}")
        cleaned_tamil = tamil_text

    # --- Stage 3: Groq LLaMA Translation (Tamil → English) -----------------
    try:
        log(f"🌐 [chunk {chunk_id}] Translating (Groq LLaMA)...")
        result = groq_translate.translate_and_extract_emotion(cleaned_tamil)

        translation = (result or {}).get("translation")
        emotion = (result or {}).get("emotion", "neutral")

        if not translation:
            log(f"⚠️  [chunk {chunk_id}] Empty translation returned — skipping.")
            return
        log(f"💬 [chunk {chunk_id}] Translation: \"{translation}\" "
            f"(emotion: {emotion})")

    except Exception as e:
        log(f"❌ [chunk {chunk_id}] Groq translation failed: {e}")
        return

    try:
        log(f"🎨 [chunk {chunk_id}] Generating emotional TTS (Cartesia)...")
        english_audio = cartesia_tts.generate_emotional_audio(
            translation, emotion
        )
        if not english_audio:
            log(f"⚠️  [chunk {chunk_id}] TTS returned no audio — skipping.")
            return
    except Exception as e:
        log(f"❌ [chunk {chunk_id}] Cartesia TTS failed: {e}")
        return

    # Hand off to the dedicated playback thread to keep audio output serialized
    playback_queue.put((chunk_id, english_audio))
    log(f"📦 [chunk {chunk_id}] Queued for playback.")


# ============================================================================
# PLAYBACK CONSUMER THREAD
# ============================================================================
def playback_worker(audio: AudioEngine, playback_queue: "queue.Queue",
                     stop_event: threading.Event, playback_active: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            chunk_id, audio_bytes = playback_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            log(f"🔊 [chunk {chunk_id}] Playing translated audio...")
            playback_active.set()
            audio.play(audio_bytes)
            playback_active.clear()
            log(f"✅ [chunk {chunk_id}] Playback finished.")
        except Exception as e:
            log(f"❌ [chunk {chunk_id}] Playback failed: {e}")
        finally:
            playback_queue.task_done()


# ============================================================================
# VAD LISTENING LOOP (main thread — must stay non-blocking on I/O)
# ============================================================================
def run_vad_loop(audio: AudioEngine, executor: ThreadPoolExecutor,
                  playback_queue: "queue.Queue", stop_event: threading.Event,
                  playback_active: threading.Event) -> None:
    vad = webrtcvad.Vad(Config.VAD_AGGRESSIVENESS)
    stream = audio.open_input_stream()

    speech_frames = deque()
    trailing_silence = 0
    in_speech = False
    chunk_id = 0

    print("\n" + "=" * 60)
    print("🎙️  Listening... (Ctrl+C to stop)")
    print("=" * 60 + "\n")

    try:
        while not stop_event.is_set():
            try:
                frame = stream.read(Config.FRAME_SIZE, exception_on_overflow=False)
            except Exception as e:
                log(f"⚠️  Mic read error: {e} — retrying...")
                time.sleep(0.05)
                continue
                
            if playback_active.is_set():
                in_speech = False
                speech_frames.clear()
                continue # Ignore microphone input while the speaker is playing

            try:
                is_speech = vad.is_speech(frame, Config.RATE)
            except Exception as e:
                log(f"⚠️  VAD error on frame: {e} — treating as silence.")
                is_speech = False

            if is_speech:
                if not in_speech:
                    in_speech = True
                    speech_frames.clear()
                    log("🎙️  Speech detected — capturing...")
                speech_frames.append(frame)
                trailing_silence = 0
            else:
                if in_speech:
                    speech_frames.append(frame)  # keep a little trailing silence
                    trailing_silence += 1

                    if trailing_silence >= Config.SILENCE_FRAMES:
                        # ---- Chunk cut point ----
                        in_speech = False
                        trailing_silence = 0
                        num_frames = len(speech_frames)

                        if num_frames >= Config.MIN_CHUNK_FRAMES:
                            chunk_bytes = b"".join(speech_frames)
                            chunk_id += 1
                            duration_ms = num_frames * Config.FRAME_DURATION_MS
                            log(f"✂️  Chunk {chunk_id} cut! "
                                f"({duration_ms} ms of audio)")

                            # Non-blocking hand-off to the worker pool
                            executor.submit(
                                process_chunk, chunk_id, chunk_bytes,
                                playback_queue
                            )
                        else:
                            log("🤏 Chunk too short — discarded as noise.")

                        speech_frames.clear()

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================
def main():
    audio = AudioEngine()
    stop_event = threading.Event()
    playback_active = threading.Event()

    def handle_sigint(signum, frame):
        log("\n🛑 Shutdown signal received — stopping AuraVox...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        playback_queue: "queue.Queue" = queue.Queue()
        executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS,
                                       thread_name_prefix="AuraVoxWorker")

        player_thread = threading.Thread(
            target=playback_worker,
            args=(audio, playback_queue, stop_event, playback_active),
            daemon=True,
        )
        player_thread.start()

        run_vad_loop(audio, executor, playback_queue, stop_event, playback_active)

    except Exception:
        log("❌ Fatal error in AuraVox orchestrator:")
        traceback.print_exc()
    finally:
        log("🧹 Cleaning up resources...")
        stop_event.set()
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        audio.close()
        log("👋 AuraVox stopped. Goodbye!")


if __name__ == "__main__":
    main()