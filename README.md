# YouTube RAG Chatbot

A conversational RAG app that turns YouTube videos into a searchable knowledge base. Paste a video, playlist, or multiple links and ask questions — answers come with clickable timestamp citations.

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

## Tech Stack

- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2`
- **Vector store:** Qdrant (in-memory)
- **Keyword search:** BM25 (`rank-bm25`)
- **Framework:** LangChain + Streamlit

## Setup

```bash
git clone https://github.com/mnkagh/youtube-chatbot.git
cd youtube-chatbot
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
