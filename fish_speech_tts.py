"""Fish Speech 1.5 TTS backend.

Runs inside a spawn subprocess — never imported in the main Flask process.
All fish_speech.* imports are inside function bodies for subprocess safety.
"""

import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

_engine = None  # Lazy singleton — first call triggers full model load


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _get_engine():
    """Load and cache the TTSInferenceEngine singleton.

    Called on first TTS request inside the subprocess. All fish_speech.*
    imports are inside this function so the Flask process never loads weights.
    """
    global _engine
    if _engine is not None:
        return _engine

    load_dotenv()

    import torch
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.vqgan.inference import load_model as load_codec
    from fish_speech.inference_engine import TTSInferenceEngine

    checkpoint_path = os.getenv("FISH_SPEECH_CHECKPOINT_PATH", "checkpoints/fish-speech-1.5")
    decoder_checkpoint = os.getenv(
        "FISH_SPEECH_DECODER_CHECKPOINT",
        "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    )
    precision_str = os.getenv("FISH_SPEECH_PRECISION", "bfloat16")
    device_str = os.getenv("FISH_SPEECH_DEVICE", "")

    # Runtime device detection when env var is not set
    if not device_str:
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"

    precision = getattr(torch, precision_str, None)
    if precision is None:
        raise ValueError(
            f"[FISH_SPEECH] Invalid FISH_SPEECH_PRECISION='{precision_str}'. "
            "Use 'float16', 'bfloat16', or 'float32'."
        )

    print(f"[INFO]    [{_ts()}] [FISH_SPEECH] Loading models: device={device_str}, precision={precision_str}")
    print(f"[INFO]    [{_ts()}] [FISH_SPEECH] Checkpoint: {checkpoint_path}")

    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device=device_str,
        precision=precision,
        compile=False,  # torch.compile unsupported on macOS; safe on all platforms
    )

    config_name = os.getenv("FISH_SPEECH_DECODER_CONFIG", "firefly_gan_vq")
    codec = load_codec(
        config_name=config_name,
        checkpoint_path=str(decoder_checkpoint),
        device=device_str,
    )

    _engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=codec,
        precision=precision,
        compile=False,
    )

    print(f"[INFO]    [{_ts()}] [FISH_SPEECH] Engine ready")
    return _engine


def generate_audio_fish(text: str) -> tuple[np.ndarray, int]:
    """Generate audio from text using Fish Speech 1.5.

    Returns:
        tuple[np.ndarray, int]: (float32 audio array, sample_rate in Hz)

    Raises:
        RuntimeError: if Fish Speech inference returns an error result.
        ValueError: if inference produces no audio chunks.
    """
    load_dotenv()

    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    ref_audio_path = os.getenv("FISH_SPEECH_REF_AUDIO", "")
    if ref_audio_path and not os.path.exists(ref_audio_path):
        print(f"[WARNING] [{_ts()}] [FISH_SPEECH] FISH_SPEECH_REF_AUDIO not found: {ref_audio_path!r} — skipping")
        ref_audio_path = ""

    references = []
    if ref_audio_path:
        with open(ref_audio_path, "rb") as f:
            ref_bytes = f.read()
        references = [ServeReferenceAudio(audio=ref_bytes, text="")]

    req = ServeTTSRequest(
        text=text,
        references=references,
        format="wav",
        seed=42,
        use_memory_cache="on",
        top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.1,
        max_new_tokens=1024,
    )

    engine = _get_engine()

    print(f"[INFO]    [{_ts()}] [FISH_SPEECH] Generating audio, chars={len(text)}")

    # Collect all audio chunks — "final" code never fires in Fish Speech 1.5;
    # instead every chunk that has audio attached is a valid piece to concatenate.
    chunks: list[np.ndarray] = []
    sample_rate = 44100

    for result in engine.inference(req):
        if result.code == "error":
            print(f"[ERROR]   [{_ts()}] [FISH_SPEECH] Inference error: {result.error}")
            raise RuntimeError(f"[FISH_SPEECH] Inference error: {result.error}")
        if result.audio is not None:
            sr, arr = result.audio
            sample_rate = sr
            chunks.append(arr)

    if not chunks:
        print(f"[ERROR]   [{_ts()}] [FISH_SPEECH] Inference returned no audio chunks")
        raise ValueError("[FISH_SPEECH] Inference returned no audio.")

    audio_array = np.concatenate(chunks, axis=-1) if len(chunks) > 1 else chunks[0]

    # Squeeze (1, N) codec output to (N,) mono before writing
    if audio_array.ndim == 2 and audio_array.shape[0] == 1:
        audio_array = audio_array.squeeze(0)

    # Normalise: int16/int32 → float32; float already in [-1,1] passes through
    if audio_array.dtype in (np.int16, np.int32):
        audio_array = audio_array.astype(np.float32) / 32768.0
    else:
        audio_array = audio_array.astype(np.float32)

    print(
        f"[INFO]    [{_ts()}] [FISH_SPEECH] Audio generated: shape={audio_array.shape}, "
        f"rate={sample_rate}, duration={audio_array.shape[-1]/sample_rate:.2f}s"
    )

    return audio_array, sample_rate
