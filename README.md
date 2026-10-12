# LLM Basics

A collection of Jupyter notebooks for learning the building blocks of large-language-model applications: API calls, LangChain, tool calling, databases, Hugging Face models, speech transcription, and summarization.

## Notebooks

| Notebook | What it covers |
| --- | --- |
| `1.Basic_Tool_Calling.ipynb` | Calling OpenAI and local Ollama models |
| `2.Langchain_Usage.ipynb` | Prompts and LangChain model integrations |
| `3.Basic_LLM_Tool_Functionality.ipynb` | Function/tool calling with a Ford inventory assistant |
| `4.Basic_LLM_DB_Operations.ipynb` | Storing and querying Ford inventory in SQLite |
| `5.Basic_LLM_Huggingface.ipynb` | Sentiment analysis, tokenization, microphone recording, Whisper transcription, and local LLM summarization |

## Setup

This project uses Python and `uv`.

```bash
uv sync
```

Start Jupyter using the project environment, then open a notebook:

```bash
uv run jupyter notebook
```

If Jupyter is not installed, add it first:

```bash
uv add jupyter
```

## Environment variables

Create a `.env` file in the project folder. Add only the keys needed for the notebook you intend to run:

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
HF_API_KEY=...
```

Keep `.env` private; never commit API keys.

## Hugging Face audio workflow

The final Hugging Face notebook records audio, saves `recording.wav`, converts it to `recording.mp3`, transcribes it with Whisper, and summarizes the transcript with a chat model.

Install FFmpeg for MP3 conversion on macOS:

```bash
brew install ffmpeg
```

The first run downloads each model, which can take time and disk space. The models are cached locally. Inspect or remove cached models with the project environment:

```bash
.venv/bin/hf cache ls
.venv/bin/hf cache rm model/HuggingFaceTB/SmolLM3-1.7B
```

## Apple Silicon / MPS

On an Apple Silicon Mac, add `device="mps"` when creating a Transformers pipeline to use the Apple GPU:

```python
pipeline("text-generation", model="HuggingFaceTB/SmolLM3-1.7B", device="mps")
```

Check availability with:

```python
import torch
print(torch.backends.mps.is_available())
```

## Data

`Test_Data/Ford.json` supplies the sample inventory used by the database and tool-calling exercises. Running the database notebook creates a local `ford_inventory.db` file.
