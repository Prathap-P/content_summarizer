import os

ASR_BACKEND = os.getenv("ASR_BACKEND", "qwen_omni")
TTS_BACKEND = os.getenv("TTS_BACKEND", "qwen_omni")
QWEN_OMNI_MODEL_ID = os.getenv("QWEN_OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni-3B")
QWEN_OMNI_SPEAKER = os.getenv("QWEN_OMNI_SPEAKER", "Chelsie")
