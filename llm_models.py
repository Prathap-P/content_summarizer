from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os


groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_completion_tokens=65000,
)

gemma_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test",
    temperature=0.7,
    model="google/gemma-3-27b"
)

nemotron_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test",
    temperature=0.3,
    model="nvidia/nemotron-3-nano",
    top_p= 0.70,
    max_completion_tokens= 10000,
    model_kwargs= {
        "frequency_penalty": 1.3, # Heavily discourages "The speaker says..." loops
        "presence_penalty": 0.3,  # Encourages introducing new topics/facts
    },
    timeout= 3600
)

nemotron_stream_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test",
    temperature=0.5,
    model="nvidia/nemotron-3-nano",
    top_p= 0.85,
    max_completion_tokens= 10000,
    model_kwargs= {
        "frequency_penalty": 1.3, # Heavily discourages "The speaker says..." loops
        "presence_penalty": 0.7,  # Encourages introducing new topics/facts
    },
    streaming=True,
    stream_usage=True,
    timeout= 3600
)

deepseekR1_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="test",
    temperature=0.7,
    model="deepseek/deepseek-r1-0528-qwen3-8b"
)

gpt_oss_20b_local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="openai/gpt-oss-20b",
    extra_body={"reasoning_effort": "high"},
    max_completion_tokens=12800,
    temperature=0.5,
    streaming=True,
)

models_collection = {
    "groq_llm": groq_llm,
    "gemma_local_llm": gemma_local_llm,
    "nemotron_local_llm": nemotron_local_llm,
    "nemotron_stream_local_llm": nemotron_stream_local_llm,
    "deepseekR1_local_llm": deepseekR1_local_llm,
    "gpt-oss_20b_local_llm": gpt_oss_20b_local_llm
}

def get_model(model_name):
    if model_name in models_collection:
        return models_collection[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")
