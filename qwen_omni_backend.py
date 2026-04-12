"""Qwen2.5-Omni backend for both ASR and TTS.

Drop-in replacements for:
  - generate_audio()             from kokoro_tts          (TTS path)
  - get_transcript_via_whisper() from whisper_transcriber  (ASR path)

The model is loaded lazily on first use so importing this module at startup
does not allocate any GPU memory.
"""

import platform
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from qwen_omni_utils import process_mm_info
except ImportError as e:
    raise ImportError("qwen-omni-utils is required for the qwen_omni backend. Run: pip install qwen-omni-utils") from e

from audio_config import QWEN_OMNI_MODEL_ID, QWEN_OMNI_SPEAKER

# Required system prompt for Qwen2.5-Omni TTS output.
_TTS_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

_DURATION_WARNING_SECONDS = 90 * 60
_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

_model = None
_processor = None


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _get_model():
    global _model, _processor
    if _model is None:
        print(f"[INFO]    [{_ts()}] [QWEN_OMNI] Loading {QWEN_OMNI_MODEL_ID} ...")
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        _processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_OMNI_MODEL_ID, use_fast=True)
        import torch
        _device_map = "auto" if not _IS_APPLE_SILICON else {"": "mps"}
        _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            QWEN_OMNI_MODEL_ID,
            torch_dtype=torch.float16 if _IS_APPLE_SILICON else "auto",
            device_map=_device_map,
        )
        print(f"[INFO]    [{_ts()}] [QWEN_OMNI] Model loaded")
    return _model, _processor


# ---------------------------------------------------------------------------
# TTS — drop-in for kokoro_tts.generate_audio()
# ---------------------------------------------------------------------------

def generate_audio_qwen(text: str) -> np.ndarray:
    """Synthesise speech from *text* and return a 24 kHz float32 numpy array."""
    model, processor = _get_model()

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": _TTS_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]

    text_input = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text_input, audios=audios, images=images, videos=videos,
        return_tensors="pt", padding=True,
    ).to(model.device)

    print(f"[INFO]    [{_ts()}] [QWEN_OMNI] Generating TTS audio (speaker={QWEN_OMNI_SPEAKER}) ...")
    try:
        out = model.generate(**inputs, use_audio_in_video=False, speaker=QWEN_OMNI_SPEAKER)
        audio_tensor = out.waveform if hasattr(out, "waveform") else out[1]
        audio_np = audio_tensor.reshape(-1).detach().cpu().numpy().astype(np.float32)
        del out, audio_tensor, inputs
    except Exception as e:
        print(f"[ERROR]   [QWEN_OMNI] TTS generation failed: {e}")
        raise
    return audio_np


# ---------------------------------------------------------------------------
# ASR — drop-in for whisper_transcriber.get_transcript_via_whisper()
# ---------------------------------------------------------------------------

def get_transcript_via_qwen(url: str, video_id: str) -> str:
    """Transcribe YouTube audio with Qwen2.5-Omni.

    Signature mirrors ``get_transcript_via_whisper(url, video_id) -> str`` so
    app.py call sites need no changes.  Returns ``"Error: ..."`` on failure.
    """
    from whisper_transcriber import download_audio, get_video_duration
    duration = get_video_duration(url)
    if duration is not None and duration > _DURATION_WARNING_SECONDS:
        mins = duration // 60
        print(f"[WARNING] [QWEN_OMNI] Video is {mins} min — transcription may be slow")

    audio_path: Path | None = download_audio(url, video_id)
    if audio_path is None:
        return "Error: audio download failed"

    model, processor = _get_model()

    conversation = [
        {
            "role": "user",
            "content": [{"type": "audio", "audio": str(audio_path)}],
        }
    ]

    try:
        text_input = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text_input, audios=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        ).to(model.device)

        print(f"[INFO]    [{_ts()}] [QWEN_OMNI] Transcribing {audio_path.name} ...")
        output = model.generate(**inputs, use_audio_in_video=False, return_audio=False)
        ids = output.sequences if hasattr(output, "sequences") else output
        decoded = processor.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        transcript = decoded[0].strip() if decoded else ""
        del output, ids, inputs
        if not transcript:
            return "Error: empty transcript from Qwen Omni"
        print(f"[INFO]    [{_ts()}] [QWEN_OMNI] Transcription complete: {len(transcript):,} chars")
        return transcript
    except Exception as e:
        print(f"[ERROR]   [QWEN_OMNI] Transcription failed: {e}")
        return f"Error: {e}"
