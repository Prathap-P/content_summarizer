import os
import copy
import threading
import time
import re
import importlib
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from dotenv import load_dotenv

load_dotenv()

VIBEVOICE_MODEL_PATH = os.getenv("VIBEVOICE_MODEL_PATH", "./models/VibeVoice-Realtime-0.5B")
VIBEVOICE_VOICES_DIR = os.getenv("VIBEVOICE_VOICES_DIR", "./VibeVoice/demo/voices/streaming_model")
VIBEVOICE_VOICE = os.getenv("VIBEVOICE_VOICE", "en-Davis_man")
def _parse_ddpm_steps() -> int:
    raw = os.getenv("VIBEVOICE_DDPM_STEPS", "30")
    try:
        return max(1, int(float(raw)))
    except ValueError:
        print(f"[WARNING] VIBEVOICE_DDPM_STEPS='{raw}' is not a valid integer — falling back to 30")
        return 30

VIBEVOICE_DDPM_STEPS: int = _parse_ddpm_steps()

class VibeVoiceTTS:
    """
    Drop-in replacement for Kokoro TTS.

    Kokoro usage (typical):
        kokoro = KPipeline(lang_code='a')
        audio, sr = kokoro(text, voice='af_heart', speed=1.0)

    VibeVoice equivalent:
        tts = VibeVoiceTTS()
        tts.load()
        audio, sr = tts.generate(text, voice='en-Emma_woman')
    """

    SAMPLE_RATE = 24_000
    CHUNK_SIZE  = 1000
    CFG_SCALE   = 1.5

    KOKORO_VOICE_MAP = {
        "af_heart":   "en-Emma_woman",
        "af_bella":   "en-Grace_woman",
        "am_adam":    "en-Carter_man",
        "am_michael": "en-Davis_man",
        "bf_emma":    "en-Emma_woman",
        "bm_george":  "en-Frank_man",
    }

    def __init__(
        self,
        model_path: str = VIBEVOICE_MODEL_PATH,
        voices_dir: str = VIBEVOICE_VOICES_DIR,
        device: str | None = None,
        ddpm_steps: int = VIBEVOICE_DDPM_STEPS,
    ):
        self.model_path = model_path
        self.voices_dir = voices_dir
        self.ddpm_steps = ddpm_steps
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
        self.model = None
        self.processor = None
        self._AudioStreamer = None
        self._voice_cache: dict = {}

    # ── Setup ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model and processor. Call once before any generate() calls."""
        from vibevoice import (
            VibeVoiceStreamingForConditionalGenerationInference,
            VibeVoiceStreamingProcessor,
        )
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn  = "flash_attention_2" if self.device == "cuda" else "sdpa"

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path, torch_dtype=dtype,
                device_map=self.device if self.device != "mps" else None,
                attn_implementation=attn,
            )
        except Exception:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path, torch_dtype=dtype,
                device_map=self.device if self.device != "mps" else None,
                attn_implementation="sdpa",
            )
        if self.device == "mps":
            self.model.to("mps")

        self.model.eval()
        try:
            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                self.model.model.noise_scheduler.config,
                algorithm_type="sde-dpmsolver++",
                beta_schedule="squaredcos_cap_v2",
            )
        except Exception:
            pass
        self.model.set_ddpm_inference_steps(num_steps=self.ddpm_steps)

        for mod_path in ("vibevoice.schedule", "vibevoice.modular", "vibevoice"):
            try:
                m = importlib.import_module(mod_path)
                if hasattr(m, "AudioStreamer"):
                    self._AudioStreamer = m.AudioStreamer
                    break
            except ImportError:
                continue
        if self._AudioStreamer is None:
            raise RuntimeError("AudioStreamer not found in vibevoice package.")

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, text: str, voice: str = "en-Carter_man", speed: float = 1.0) -> tuple[np.ndarray, int]:
        """
        Generate audio from text.

        Args:
            text  : input text (any length — auto-chunked if >CHUNK_SIZE chars)
            voice : built-in voice name (e.g. 'en-Carter_man') OR
                    Kokoro voice name (auto-mapped, e.g. 'af_heart') OR
                    path to a .pt preset file OR
                    path to a reference .wav for cloning
            speed : ignored (VibeVoice does not expose speed control directly)

        Returns:
            (audio: np.ndarray[float32], sample_rate: int)
        """
        if self.model is None:
            raise RuntimeError("Call .load() before .generate()")

        prefilled = self._resolve_voice(voice)
        text = self._normalise_text(text)

        chunks = self._chunk_text(text)
        all_audio = []
        for chunk in chunks:
            audio, _ = self._synthesize_one(chunk, prefilled)
            all_audio.append(audio)

        final = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
        return final, self.SAMPLE_RATE

    def save(self, audio: np.ndarray, path: str) -> None:
        """Save float32 audio array to 16-bit PCM WAV file."""
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(path, self.SAMPLE_RATE, pcm)

    @property
    def available_voices(self) -> list:
        """Return names of all available built-in voice presets."""
        if not os.path.isdir(self.voices_dir):
            return []
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self.voices_dir)
            if f.endswith(".pt")
        ]

    # ── Internals ────────────────────────────────────────────────────────────

    def _resolve_voice(self, voice: str):
        """Resolve voice string → prefilled_outputs (KV cache tensor)."""
        if voice in self._voice_cache:
            return self._voice_cache[voice]

        mapped = self.KOKORO_VOICE_MAP.get(voice, voice)

        if mapped.endswith(".pt") and os.path.exists(mapped):
            pt_path = mapped
        else:
            pt_path = os.path.join(self.voices_dir, f"{mapped}.pt")

        if os.path.exists(pt_path):
            # weights_only=False required: .pt files are trusted KV-cache tensors, not user-uploaded
            prefilled = torch.load(pt_path, map_location=torch.device(self.device), weights_only=False)
            self._voice_cache[voice] = prefilled
            return prefilled

        if os.path.exists(mapped) and mapped.endswith(".wav"):
            prefilled = self._clone_from_wav(mapped)
            self._voice_cache[voice] = prefilled
            return prefilled

        fallback = os.path.join(self.voices_dir, "en-Carter_man.pt")
        if os.path.exists(fallback):
            print(f"[WARNING] [VIBEVOICE_TTS] Voice '{voice}' not found — using en-Carter_man")
            # weights_only=False required: .pt files are trusted KV-cache tensors, not user-uploaded
            prefilled = torch.load(fallback, map_location=torch.device(self.device), weights_only=False)
            self._voice_cache[voice] = prefilled
            return prefilled

        raise FileNotFoundError(f"No voice preset found for '{voice}' in {self.voices_dir}")

    def _clone_from_wav(self, wav_path: str):
        """Best-effort voice clone from reference WAV → KV cache."""
        try:
            import librosa
            audio_arr, _ = librosa.load(wav_path, sr=self.SAMPLE_RATE, mono=True)
            audio_tensor = torch.tensor(audio_arr, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                inputs = self.processor(
                    audios=audio_tensor, sampling_rate=self.SAMPLE_RATE,
                    return_tensors="pt", padding=True,
                )
                inputs = {k: v.to(torch.device(self.device)) if hasattr(v, "to") else v
                          for k, v in inputs.items()}
                outputs = self.model(**inputs, use_cache=True)
                return outputs.past_key_values
        except Exception as e:
            raise RuntimeError(f"Voice cloning from {wav_path} failed: {e}")

    def _normalise_text(self, text: str) -> str:
        return (text.replace("\u2018", "'").replace("\u2019", "'")
                    .replace("\u201c", '"').replace("\u201d", '"'))

    def _chunk_text(self, text: str) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 1 <= self.CHUNK_SIZE:
                current = (current + " " + s).strip()
            else:
                if current:
                    chunks.append(current)
                while len(s) > self.CHUNK_SIZE:
                    chunks.append(s[:self.CHUNK_SIZE])
                    s = s[self.CHUNK_SIZE:]
                current = s
        if current:
            chunks.append(current)
        return chunks or [text]

    def _synthesize_one(self, text: str, prefilled_outputs) -> tuple[np.ndarray, float]:
        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip(), cached_prompt=prefilled_outputs,
            padding=True, return_tensors="pt", return_attention_mask=True,
        )
        inputs = {k: v.to(torch.device(self.device)) if hasattr(v, "to") else v
                  for k, v in processed.items()}

        streamer   = self._AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_event = threading.Event()
        ttfa_s: list = []
        t_start    = time.perf_counter()

        def _run() -> None:
            try:
                self.model.generate(
                    **inputs, max_new_tokens=None, cfg_scale=self.CFG_SCALE,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False, "temperature": 1.0, "top_p": 1.0},
                    audio_streamer=streamer, stop_check_fn=stop_event.is_set,
                    verbose=False, refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                )
            except Exception as exc:
                errors.append(exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        chunks = []
        try:
            for chunk in streamer.get_stream(0):
                if not ttfa_s:
                    ttfa_s.append(time.perf_counter() - t_start)
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)
                peak = float(np.max(np.abs(chunk)))
                if peak > 1.0:
                    chunk = chunk / peak
                chunks.append(chunk)
        finally:
            stop_event.set()
            streamer.end()
            thread.join()

        if errors:
            raise errors[0]

        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        return audio, (ttfa_s[0] if ttfa_s else 0.0)

# ── Module-level lazy singleton ──────────────────────────────────────────────

_tts: VibeVoiceTTS | None = None

def generate_audio(text: str, voice: str = "") -> tuple[np.ndarray, int]:
    """Generate TTS audio using VibeVoice.

    Lazy-loads the model on first call. Returns (audio_array, sample_rate).
    voice: override voice name; uses VIBEVOICE_VOICE env var default if empty.
    """
    global _tts
    if _tts is None:
        print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] [VIBEVOICE_TTS] Loading model from {VIBEVOICE_MODEL_PATH}...")
        _tts = VibeVoiceTTS(
            model_path=VIBEVOICE_MODEL_PATH,
            voices_dir=VIBEVOICE_VOICES_DIR,
            device=None,  # auto-detect
            ddpm_steps=VIBEVOICE_DDPM_STEPS,
        )
        _tts.load()
        print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] [VIBEVOICE_TTS] Model loaded. Available voices: {_tts.available_voices}")

    resolved_voice = voice if voice else VIBEVOICE_VOICE
    print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] [VIBEVOICE_TTS] Generating audio for {len(text)} chars, voice={resolved_voice!r}")
    audio, sample_rate = _tts.generate(text, voice=resolved_voice)
    print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] [VIBEVOICE_TTS] Audio generated: {len(audio)/sample_rate:.2f}s at {sample_rate}Hz")
    return audio, sample_rate
