# YOUBO

Chat with YouTube videos.

## Features

- **Timestamped answers** — every answer links back to the exact moment in the video (`&t=` seconds)
- **Hybrid retrieval** — Qdrant vector search (dense) + BM25 keyword search, merged and deduplicated
- **Chat history** — follow-up questions work via history-aware retrieval
- **Multi-video support** — chat across several videos at once
- **Comparison mode** — explicitly compare what different videos say about a topic
- **Playlist support** — paste a playlist URL, it expands to the first 10 videos
- **Summary levels** — TL;DR, Short, or Detailed summaries
- **Quiz generator** — auto-creates multiple-choice quizzes from a video with answers + explanations
- **Study notes export** — structured markdown notes, downloadable as `.md` or `.pdf`
- **Key moments timeline** — evenly-spaced timestamped highlights with links
- **Video metadata** — titles, channels, durations shown in the sidebar
- **Chat export** — download the full conversation as markdown
- **Cached index** — the vector/BM25 index is built once per video set, not on every rerun
- **Input hardening** — length limits, playlist caps, transcript size caps, graceful errors
- **Server logging** — every failure is logged with timings to `logs/app.log`, viewable in the sidebar Diagnostics panel

## Tech Stack

- **LLM:** your choice of free provider (see below)
- **Embeddings:** FastEmbed `bge-small-en-v1.5` (ONNX — loads in <1s on CPU, no torch)
- **Vector store:** Qdrant (in-memory)
- **Keyword search:** BM25 (`rank-bm25`)
- **Framework:** LangChain + Streamlit

## LLM Providers (all free)

Pick one in `.env` at runtime — no code change needed, and visitors never see it:

```bash
LLM_PROVIDER=Ollama (local, unlimited, no key)  # default, keyless
# LLM_PROVIDER=Groq (free tier: 30 rpm / 1k day)  # + GROQ_API_KEY
```

Resolution order: configured provider with key → working Ollama → anonymous Pollinations fallback.

| Provider | Model default | Free allowance | Key |
|----------|---------------|----------------|-----|
| **Ollama** (local) | `llama3.2` | Unlimited, fully offline | None |
| **Pollinations** | `openai` | Anonymous, no signup — but rate-limited, best as fallback | None |
| **LM Studio / llama.cpp** (local server) | `local-model` | Unlimited, runs your own GGUF models | None |
| **Groq** | `openai/gpt-oss-120b` | 30 req/min, 1000/day, 200k tokens/day | [console.groq.com/keys](https://console.groq.com/keys) |
| **Google Gemini** | `gemini-3-flash-preview` | Free Flash models, per-project caps | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| **Hugging Face** | `Llama-3.1-8B-Instruct` | Free tier with free account | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **OpenRouter** | `openai/gpt-oss-120b:free` | Free models, ~50 req/day | [openrouter.ai/keys](https://openrouter.ai/keys) |

**For the long term:** Ollama is the most sustainable — no key, no quota, no data leaving your machine, no vendor deprecating your model. Groq's free tier is the best documented cloud option (published limits, no card). Free tiers change often: Groq replaced its Llama free models with `gpt-oss`, and Cerebras/GitHub Models/Together dropped free tiers in 2026 — hence the provider is configurable rather than hardcoded.

To go fully local:

```bash
ollama pull llama3.2
```

## Keyless for end users

Only the **host** machine ever needs Ollama or a key — visitors just open a URL:

1. **Local (Ollama default):** install Ollama once on the host, run `ollama pull llama3.2`, select the Ollama provider. No visitor ever enters a key.
2. **Server key:** set `GROQ_API_KEY` (or Gemini/OpenRouter) once in `.env` on the host. The app detects it and hides the key field from visitors.
3. **Docker (host it anywhere):**

```bash
docker compose up --build -d
docker compose exec ollama ollama pull llama3.2
```

Then share `http://<host>:8501` — fully keyless for everyone.

## Setup

```bash
git clone https://github.com/mnkagh/YOUBO.git
cd YOUBO
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create a `.env` file (or paste your key in the UI):

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=optional_huggingface_token
```

Run:

```bash
streamlit run app.py
```

## Usage

1. Enter your Groq API key
2. Paste YouTube URL(s) or a playlist in the sidebar and click **Load videos**
3. Ask questions in the chat — answers include clickable timestamp links
4. Use comparison mode for multi-video analysis, or generate/export study notes

## How It Works

1. Transcript fetched via `youtube-transcript-api` (with timestamps)
2. Chunked into ~1200-char segments, each tagged with its video + start time
3. Embedded and stored in Qdrant; BM25 index built in parallel
4. Questions are rewritten using chat history, then answered via hybrid retrieval with `[Source](link)` citations

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key (or enter in UI) |
| `HF_TOKEN` | No | Hugging Face token |

## License

MIT
