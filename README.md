# Website Summarizer with TTS

A technical news article explainer that fetches web content, provides intelligent summaries and explanations using LangChain and Groq LLM, and converts responses to audio using Kokoro TTS.

## Features

- **Web Content Extraction**: Fetch and parse news articles from URLs using `read_website_content`
- **Conversational AI**: Interactive Q&A about technical articles with context retention using `ConversationChain`
- **Text-to-Speech**: Generate natural-sounding audio from responses using `generate_and_create_audio_file`
- **TTS-Optimized Output**: LLM responses formatted specifically for spoken audio (no markdown, expanded acronyms, natural number reading)
- **Rich Terminal Display**: Formatted markdown output in terminal using Rich library

## Project Structure

```
.
├── main.py                 # URL content loader
├── news_reader.py          # Main conversational interface (Python script)
├── news_reader.ipynb       # Jupyter notebook version
├── kokoro_tts.py          # Text-to-speech generation
├── requirements.txt        # Python dependencies
├── .env                   # API keys (not in version control)
├── kokoro_outputs/        # Generated audio files
└── __pycache__/          # Python cache
```

## Setup

### Prerequisites

- Python 3.11+
- GROQ API key

### Installation

1. Clone the repository

2. Create and activate a virtual environment:
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```sh
pip install -r requirements.txt
```

4. Create a `.env` file with your API key:
```
GROQ_API_KEY=your_key_here
```

## Email Configuration

To enable email sending functionality:

1. **Generate Gmail App Password:**
   - Visit https://myaccount.google.com/apppasswords
   - Create an app password for "Mail"
   - Copy the 16-character password

2. **Set Environment Variables:**
   ```bash
   export GMAIL_ADDRESS='your-email@gmail.com'
   export GMAIL_APP_PASSWORD='your-16-char-app-password'
   export RECIPIENT_GMAIL_ADDRESS='recipient-email@gmail.com'
   ```

3. **Add to Shell Profile (Optional):**
   Add the above exports to your `~/.zshrc` or `~/.bashrc` to make them permanent.

## Usage

### Command-Line Interface

Run the main conversation loop:

```sh
python news_reader.py
```

Then paste a news article URL or content when prompted. The system will:
1. Analyze the article
2. Provide an initial explanation
3. Allow follow-up questions
4. Generate audio for each response in `kokoro_outputs`

Type `exit` or `quit` to end the conversation.

### Jupyter Notebook

Open and run `news_reader.ipynb` for an interactive notebook experience with inline outputs.

## Key Components

### `main.py`
- `read_website_content(url)`: Extracts article content using `NewsURLLoader`

### `news_reader.py`
- `groq_llm`: ChatGroq instance configured with `openai/gpt-oss-20b` model
- `conversation_chain`: LangChain conversation chain with 100-turn memory window
- `system_message_news_explainer`: Custom system prompt optimized for TTS output

### `kokoro_tts.py`
- `generate_audio(text)`: Converts text to audio using Kokoro pipeline
- `create_audio_file(audio)`: Saves audio with timestamp to `kokoro_outputs`
- `generate_and_create_audio_file(text)`: Combined generation and saving

## Configuration

### LLM Settings
Adjust in `news_reader.py`:
- `model`: LLM model name
- `temperature`: 0.3 (controls randomness)
- `max_tokens`: 1000 (response length limit)

### TTS Settings
In `kokoro_tts.py`:
- `lang_code`: 'a' (American English)
- `voice`: 'af_heart'
- `sr`: 24000 (sample rate)

### Memory
- `ConversationBufferWindowMemory` retains last 100 conversation turns

## Output

Audio files are saved to `kokoro_outputs` with format:
```
kokoro_500word_YYYYMMDD_HHMMSS.wav
```

## Notes

- The system prompt includes **TTS-SAFE OUTPUT RULES** that prevent markdown, code blocks, and other non-speakable formats
- Responses are optimized for natural speech (expanded acronyms, readable numbers)
- GPU acceleration is automatically used if CUDA is available
- Conversation memory is cleared on exit

## Dependencies

Key packages (see `requirements.txt` for full list):
- `langchain` / `langchain-groq` - LLM orchestration
- `kokoro` - Text-to-speech
- `newspaper3k` - Web content extraction
- `rich` - Terminal formatting
- `torch` - Deep learning backend

## License

See repository for license information.
