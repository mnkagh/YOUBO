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
- **No-captions fallback** — caption-less videos are transcribed automatically
- **Chat export** — download the full conversation as markdown
- **Cached index** — the vector/BM25 index is built once per video set, not on every rerun
- **Input hardening** — length limits, playlist caps, transcript size caps, graceful errors
- **Server logging** — every failure is logged with timings to `logs/app.log`, viewable in the sidebar Diagnostics panel

## Tech Stack

- **LLM:** Groq (`openai/gpt-oss-120b`)
- **Embeddings:** FastEmbed `bge-small-en-v1.5` (ONNX — loads in <1s on CPU, no torch)
- **Vector store:** Qdrant (in-memory)
- **Keyword search:** BM25 (`rank-bm25`)
- **Framework:** LangChain + Streamlit

## Setup

```bash
git clone https://github.com/mnkagh/YOUBO.git
cd YOUBO
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Usage

1. Paste YouTube URL(s) or a playlist in the sidebar and click **Load videos**
2. Ask questions in the chat — answers include clickable timestamp links
3. Use comparison mode for multi-video analysis, or generate/export study notes
4. Use the Dub tab to translate or dub the video into another language

## How It Works

1. Transcript fetched via `youtube-transcript-api` (with timestamps)
2. Chunked into ~1200-char segments, each tagged with its video + start time
3. Embedded and stored in Qdrant; BM25 index built in parallel
4. Questions are rewritten using chat history, then answered via hybrid retrieval with `[Source](link)` citations

## License

MIT
