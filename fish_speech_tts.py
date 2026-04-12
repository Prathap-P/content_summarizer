"""Fish Speech S1-mini TTS backend.

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
    from fish_speech.models.vqgan.inference import load_model as load_decoder_model
    from fish_speech.inference_engine import TTSInferenceEngine

    checkpoint_path = os.getenv("FISH_SPEECH_CHECKPOINT_PATH", "checkpoints/s1-mini")
    decoder_config = os.getenv("FISH_SPEECH_DECODER_CONFIG", "firefly_gan_vq")
    decoder_checkpoint = os.getenv(
        "FISH_SPEECH_DECODER_CHECKPOINT",
        "checkpoints/s1-mini/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
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

    decoder_model = load_decoder_model(
        config_name=decoder_config,
        checkpoint_path=decoder_checkpoint,
        device=device_str,
    )

    _engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=False,
    )

    print(f"[INFO]    [{_ts()}] [FISH_SPEECH] Engine ready")
    return _engine


def generate_audio_fish(text: str) -> tuple[np.ndarray, int]:
    """Generate audio from text using Fish Speech S1-mini.

    Returns:
        tuple[np.ndarray, int]: (float32 audio array, sample_rate in Hz)

    Raises:
        RuntimeError: if Fish Speech inference returns an error result.
    """
    load_dotenv()

    from fish_speech.utils.schema import ServeTTSRequest

    ref_audio_path = os.getenv("FISH_SPEECH_REF_AUDIO", "")

    engine = _get_engine()

    req_kwargs = dict(text=text, streaming=False)
    if ref_audio_path:
        from fish_speech.inference_engine.reference_loader import ServeReferenceAudio
        with open(ref_audio_path, "rb") as f:
            ref_bytes = f.read()
        req_kwargs["references"] = [ServeReferenceAudio(audio=ref_bytes, text="")]

    req = ServeTTSRequest(**req_kwargs)

    print(f"[INFO]    [{_ts()}] [FISH_SPEECH] Generating audio, chars={len(text)}")

    audio_array = None

    for result in engine.inference(req):
        if result.code == "error":
            print(f"[ERROR]   [{_ts()}] [FISH_SPEECH] Inference error: {result.error}")
            raise RuntimeError(f"[FISH_SPEECH] Inference error: {result.error}")
        if result.code == "final":
            sample_rate, audio_array = result.audio
            break

    if audio_array is None:
        print(f"[ERROR]   [{_ts()}] [FISH_SPEECH] No audio produced — inference returned no final result")
        raise RuntimeError("[FISH_SPEECH] No audio produced — inference returned no final result")

    if audio_array.size == 0:
        print(f"[ERROR]   [{_ts()}] [FISH_SPEECH] Inference returned an empty audio array")
        raise RuntimeError("[FISH_SPEECH] Inference returned an empty audio array")

    # Ensure float32 (Fish Speech may return int16 or other dtypes)
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if audio_array.max() > 1.0 or audio_array.min() < -1.0:
        audio_array = audio_array / 32768.0

    print(
        f"[INFO]    [{_ts()}] [FISH_SPEECH] Audio generated: shape={audio_array.shape}, "
        f"rate={sample_rate}, duration={len(audio_array)/sample_rate:.2f}s"
    )

    return audio_array, sample_rate
