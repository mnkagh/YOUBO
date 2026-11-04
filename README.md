# YOUBO

Paste a YouTube link, ask questions about the video. Answers point back to the
exact moment they came from.

## What it does

- Chat with one video or a playlist (first 10 videos per playlist)
- Summaries (TL;DR / short / detailed), study notes, multiple-choice quizzes
- Transcript translation and video dubbing
- Clickable timestamps everywhere, so you can check every answer yourself
- Optional accounts, or use it as a guest

## How it works

Transcripts come from `youtube-transcript-api` and get split into
~1200-character chunks, each tagged with its video and start time. Questions
are answered with hybrid retrieval (Qdrant vector search for meaning, BM25 for
exact names and keywords) through a history-aware chain, so follow-up
questions still make sense. Videos without captions fall back to audio
transcription where possible.

## Running it

```bash
git clone https://github.com/mnkagh/YOUBO.git
cd YOUBO
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Either install Ollama or put a free key in `.env`:

```
LLM_PROVIDER=Groq (free tier: 30 rpm / 1k day)
GROQ_API_KEY=your key
```

Other providers work too (Gemini, OpenRouter, LM Studio, llama.cpp) — set
`LLM_PROVIDER` to the matching option in `providers.py`.

## Notes

- Live demo: https://youbo.onrender.com/
- Chat history and app files stay on the server, nothing is shared elsewhere.

## License

MIT
