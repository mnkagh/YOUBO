"""Core logic for YOUBO: loading, retrieval, generation helpers."""
import logging
import os
import re
import time
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

MAX_VIDEOS = 20
MAX_TRANSCRIPT_CHARS = 200_000

LOG_DIR = Path(__file__).parent / "logs"


def get_logger(name: str = "ytchat") -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


def log_error(action: str, exc: Exception) -> None:
    log.exception(f"{action} failed: {exc}")


def read_recent_logs(n: int = 40) -> str:
    try:
        lines = (LOG_DIR / "app.log").read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-n:]) or "(no log entries yet)"
    except FileNotFoundError:
        return "(no log entries yet)"


class Timer:
    def __init__(self, action: str):
        self.action = action

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        log.info(f"{self.action} took {time.time() - self.t0:.1f}s")


_EMBEDDING = None


def get_embeddings():
    """Process-level singleton: the model loads once, not on every rerun.

    FastEmbed (ONNX) instead of sentence-transformers (torch): loads in
    seconds on CPU instead of ~40s, no GPU/torch needed.
    """
    global _EMBEDDING
    if _EMBEDDING is None:
        with Timer("embedding model load"):
            from langchain_community.embeddings import FastEmbedEmbeddings
            _EMBEDDING = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _EMBEDDING

PROVIDERS = {
    "Ollama (local, unlimited, no key)": {"model": "llama3.2", "env": "", "needs_key": False},
    "Pollinations (anonymous, rate-limited)": {
        "model": "openai", "env": "", "needs_key": False,
        "base_url": "https://text.pollinations.ai/openai",
    },
    "LM Studio (local server, no key)": {
        "model": "local-model", "env": "", "needs_key": False,
        "base_url": "http://localhost:1234/v1",
    },
    "llama.cpp (local server, no key)": {
        "model": "default", "env": "", "needs_key": False,
        "base_url": "http://localhost:8080/v1",
    },
    "Hugging Face (free tier)": {"model": "meta-llama/Llama-3.1-8B-Instruct", "env": "HF_TOKEN", "needs_key": True},
    "Groq (free tier: 30 rpm / 1k day)": {"model": "openai/gpt-oss-120b", "env": "GROQ_API_KEY", "needs_key": True},
    "Google Gemini (free Flash models)": {"model": "gemini-3-flash-preview", "env": "GEMINI_API_KEY", "needs_key": True},
    "OpenRouter (free models, 50/day)": {
        "model": "openai/gpt-oss-120b:free", "env": "OPENROUTER_API_KEY", "needs_key": True,
        "base_url": "https://openrouter.ai/api/v1",
    },
}

PROVIDER_LINKS = {
    "Groq (free tier: 30 rpm / 1k day)": "https://console.groq.com/keys",
    "Google Gemini (free Flash models)": "https://aistudio.google.com/app/apikey",
    "OpenRouter (free models, 50/day)": "https://openrouter.ai/keys",
    "Hugging Face (free tier)": "https://huggingface.co/settings/tokens",
    "Ollama (local, unlimited, no key)": "https://ollama.com/download",
    "LM Studio (local server, no key)": "https://lmstudio.ai",
    "llama.cpp (local server, no key)": "https://github.com/ggerganov/llama.cpp",
    "Pollinations (anonymous, rate-limited)": "https://pollinations.ai",
}


def ollama_status(model: str = "llama3.2") -> dict:
    """Ping the local Ollama server. Host-only concern; end users never see this."""
    import requests
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        models = [m.get("name", "") for m in r.json().get("models", [])]
        return {"running": True, "has_model": any(model in m for m in models), "models": models}
    except Exception:
        return {"running": False, "has_model": False, "models": []}


def resolve_provider() -> tuple:
    """Host-side config, invisible to end users.

    Priority: LLM_PROVIDER env (+ its key env) -> working Ollama -> Pollinations fallback.
    Returns (provider_name, model, api_key).
    """
    configured = os.getenv("LLM_PROVIDER", "Ollama (local, unlimited, no key)")
    if configured not in PROVIDERS:
        configured = "Ollama (local, unlimited, no key)"
    model = os.getenv("LLM_MODEL", "") or PROVIDERS[configured]["model"]

    if not configured.startswith("Ollama"):
        key = os.getenv(PROVIDERS[configured]["env"], "")
        if key:
            return configured, model, key
        # Configured provider has no key -> fall through to keyless options.

    status = ollama_status(model if configured.startswith("Ollama") else PROVIDERS["Ollama (local, unlimited, no key)"]["model"])
    if status["running"] and status["has_model"]:
        name = "Ollama (local, unlimited, no key)"
        return name, PROVIDERS[name]["model"], ""

    return (
        "Pollinations (anonymous, rate-limited)",
        PROVIDERS["Pollinations (anonymous, rate-limited)"]["model"],
        "",
    )


def get_llm(provider: str, api_key: str, model: str | None = None):
    """Build a chat model for the chosen provider. Every option above works on a free tier."""
    if PROVIDERS[provider]["needs_key"] and not api_key:
        raise ValueError(f"{provider} needs an API key. Get a free one at {PROVIDER_LINKS[provider]}")
    model = model or PROVIDERS[provider]["model"]

    if provider.startswith("Ollama"):
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model, temperature=0.2,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )

    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.2)

    if provider.startswith("Google Gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)

    if provider.startswith("Hugging Face"):
        return HuggingFaceChat(model=model, api_key=api_key, temperature=0.2)

    if "base_url" in PROVIDERS[provider]:
        return OpenAICompatibleChat(
            model=model, api_key=api_key, temperature=0.2,
            base_url=PROVIDERS[provider]["base_url"],
        )

    raise ValueError(f"Unsupported provider: {provider}")


class OpenAICompatibleChat(BaseChatModel):
    """Any OpenAI-compatible endpoint (OpenRouter, Pollinations, LM Studio,
    llama.cpp, vLLM). Plain requests, no extra dependencies."""

    model: str
    api_key: str = ""
    temperature: float = 0.2
    base_url: str = ""

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def _to_openai_role(self, m) -> str:
        return {"human": "user", "ai": "assistant"}.get(m.type, m.type)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": self._to_openai_role(m), "content": m.content if isinstance(m.content, str) else str(m.content)}
                for m in messages
            ],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=120,
                )
            except Exception as e:
                last_error = f"connection failed (attempt {attempt + 1}/3): {e}"
                log.warning(last_error)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=resp.json()["choices"][0]["message"]["content"]))]
                )
            last_error = f"LLM error {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                log.warning(f"{last_error} — retrying")
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_error)
        raise RuntimeError(last_error)


class HuggingFaceChat(BaseChatModel):
    """Hugging Face Inference API (free tier with a free account token)."""

    model: str
    api_key: str
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "huggingface"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from huggingface_hub import InferenceClient
        from langchain_core.messages import SystemMessage
        client = InferenceClient(token=self.api_key)
        out = client.chat.completions.create(
            model=self.model,
            messages=[
                ({"role": "system" if isinstance(m, SystemMessage) else m.type, "content": str(m.content)})
                for m in messages
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=out.choices[0].message.content))]
        )


# Backwards-compat alias (was OpenRouter-only before generalization).
OpenRouterChat = OpenAICompatibleChat


def extract_video_id(text: str) -> str | None:
    text = text.strip()[:500]
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([\w-]{11})", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[\w-]{11}", text):
        return text
    return None


LANGUAGES = {
    "Auto (any available)": "auto",
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
    "German": "de", "Portuguese": "pt", "Arabic": "ar", "Russian": "ru",
    "Tamil": "ta", "Telugu": "te", "Bengali": "bn", "Marathi": "mr",
    "Japanese": "ja", "Korean": "ko",
}


def is_playlist_url(text: str) -> bool:
    return "list=" in text


def expand_playlist(url: str, limit: int = 10) -> list[str]:
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return [(e.get("url") or e.get("id")) for e in info.get("entries", [])[:limit] if e]
    except Exception:
        return []


def get_video_metadata(video_id: str) -> dict:
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        return {
            "title": info.get("title", video_id),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "unknown"),
            "view_count": info.get("view_count"),
            "thumbnail": info.get("thumbnail"),
            "categories": info.get("categories") or [],
        }
    except Exception:
        return {"title": video_id, "duration": 0, "uploader": "unknown",
                "view_count": None, "thumbnail": None, "categories": []}


def content_type_of(meta: dict) -> str:
    cats = [c.lower() for c in (meta.get("categories") or [])]
    joined = " ".join(cats)
    if "music" in joined:
        return "music"
    if "gaming" in joined:
        return "gaming"
    if any(k in joined for k in ("education", "science", "howto", "news", "documentary", "history")):
        return "educational"
    return "general"

def fetch_transcript(video_id: str, language: str = "auto") -> tuple:
    """Returns (snippets, language_name). Prefers the requested language,
    falls back to whatever captions exist (manual first, then auto-generated)."""
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    listing = api.list(video_id)
    available = list(listing)
    transcript = None
    if language != "auto":
        try:
            transcript = listing.find_transcript([language])
        except Exception:
            transcript = None
    if transcript is None:
        manual = [t for t in available if not t.is_generated]
        transcript = (manual or available or [None])[0]
    if transcript is None:
        raise RuntimeError("No captions exist for this video.")
    data = transcript.fetch()
    snippets = [{"text": s.text, "start": s.start, "duration": s.duration} for s in data]
    return snippets, transcript.language


def transcript_to_docs(video_id: str, snippets: list[dict]) -> list[Document]:
    chunks, current, size = [], [], 0
    start = snippets[0]["start"] if snippets else 0
    total = 0
    for s in snippets:
        current.append(s["text"])
        size += len(s["text"])
        total += len(s["text"])
        if total > MAX_TRANSCRIPT_CHARS:
            break
        if size >= 1200:
            chunks.append(_make_chunk(video_id, " ".join(current), start))
            current, size = [], 0
            start = s["start"] + s.get("duration", 0)
    if current:
        chunks.append(_make_chunk(video_id, " ".join(current), start))
    return chunks


def _make_chunk(video_id: str, text: str, start: float) -> Document:
    return Document(
        page_content=text,
        metadata={
            "video_id": video_id,
            "start": start,
            "link": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
        },
    )


def build_hybrid_retriever(video_ids: list, docs: list):
    """Vector (Qdrant) + BM25 hybrid retrieval resources."""
    embedding = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = []
    for d in docs:
        splits.extend(splitter.split_documents([d]))
    if not splits:
        splits = list(docs)
    vectorstore = QdrantVectorStore.from_documents(
        splits, embedding, location=":memory:", collection_name="yt_chat"
    )
    bm25 = BM25Okapi([d.page_content.lower().split() for d in splits])
    return vectorstore, splits, bm25


class HybridRetriever(BaseRetriever):
    vectorstore: object = None
    splits: list = []
    bm25: object = None
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        vec = vector_retriever.invoke(query)
        scores = self.bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        kw = [self.splits[i] for i in top if scores[i] > 0]
        seen, merged = set(), []
        for d in list(vec) + kw:
            key = d.page_content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(d)
            if len(merged) >= self.k:
                break
        return merged


class LinkAttachingRetriever(BaseRetriever):
    base: object = None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        out = []
        for d in self.base.invoke(query):
            link = d.metadata.get("link")
            out.append(Document(
                page_content=f"{d.page_content}\n[Source]({link})" if link else d.page_content,
                metadata=d.metadata,
            ))
        return out


def build_notes(llm, transcript_text: str, content_type: str = "general", max_chars: int = 12000) -> str:
    style = {
        "music": "Turn this music video transcript/lyrics into fan-style notes: themes, standout lines, and vibe, in markdown.",
        "gaming": "Turn this gaming video transcript into notes: key plays, strategies, and highlights, in markdown.",
        "educational": "Turn this transcript into clean, structured study notes in markdown with headings, bullet points, and key takeaways.",
        "general": "Turn this video transcript into clean, structured notes in markdown with headings, bullet points, and key takeaways.",
    }[content_type]
    return llm.invoke(f"{style}\n\n{transcript_text[:max_chars]}").content


def build_summary(llm, transcript_text: str, level: str, content_type: str = "general", max_chars: int = 12000) -> str:
    instructions = {
        "TL;DR": "Write a TL;DR summary in at most 3 sentences.",
        "Short": "Write a short summary in one paragraph (~5 sentences).",
        "Detailed": "Write a detailed summary with sections: Overview, Main Points, Conclusion.",
    }
    hint = {"music": "Focus on theme, mood, and message. ",
            "gaming": "Focus on what happens and key moments. ",
            "educational": "Focus on concepts taught. ",
            "general": ""}[content_type]
    prompt = f"{hint}{instructions.get(level, instructions['Short'])}\n\n{transcript_text[:max_chars]}"
    return llm.invoke(prompt).content


def build_quiz(llm, transcript_text: str, n: int = 5, max_chars: int = 12000) -> str:
    prompt = (
        f"Based on this YouTube transcript, create a quiz with {n} multiple-choice questions. "
        "Each question must have options A-D, the correct answer marked, and a one-line explanation. "
        "Format in markdown.\n\n" + transcript_text[:max_chars]
    )
    return llm.invoke(prompt).content


def build_quiz_data(llm, transcript_text: str, content_type: str = "general", n: int = 5, max_chars: int = 12000) -> list:
    """Structured quiz for the interactive quiz UI. Returns a list of
    {question, options[4], answer (0-3), explanation}."""
    import json
    flavor = {
        "music": "fun trivia about the song, artist, lyrics and theme",
        "gaming": "questions about plays, strategies and moments in the video",
        "educational": "questions testing understanding of the concepts taught",
        "general": "questions about the key points of the video",
    }[content_type]
    prompt = (
        f"Based on the transcript below, write {n} multiple-choice questions ({flavor}). "
        "Reply with ONLY a valid JSON array, no markdown fences, no commentary. "
        'Each item: {"question": "...", "options": ["...", "...", "...", "..."], '
        '"answer": 0-3 index of the correct option, "explanation": "one line"}.\n\n'
        + transcript_text[:max_chars]
    )
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    clean = []
    for item in data[:n]:
        opts = [str(o) for o in item.get("options", [])][:4]
        ans = int(item.get("answer", 0))
        if len(opts) == 4 and 0 <= ans <= 3 and item.get("question"):
            clean.append({
                "question": str(item["question"]),
                "options": opts,
                "answer": ans,
                "explanation": str(item.get("explanation", "")),
            })
    if not clean:
        raise ValueError("Quiz came back empty — try again.")
    return clean


GUIDE_MD = """
## How YOUBO works — the honest version

**The pipeline:** YouTube captions → text chunks → embeddings → Qdrant (vectors) + BM25 (keywords) → history-aware retrieval → LLM answer with timestamp links.

### What each tool does, and what you could use instead

| Tool | Job here | Alternatives | Pros of ours | Cons of ours |
|---|---|---|---|---|
| **Streamlit** | The web UI | Gradio, Next.js, Flask | 5-minute UIs in pure Python | Reruns whole script per click; not ideal for huge scale |
| **LangChain** | Wires retriever + LLM + history | LlamaIndex, raw API calls | Standard building blocks | Heavy dependency, APIs shift fast (v1 moved `chains`) |
| **Qdrant** | Vector search over chunks | FAISS, Chroma, Pinecone | Fast, filterable, runs in-memory with zero setup | In-memory = rebuilt per load (fine for <50 videos) |
| **BM25** | Keyword search (names, exact terms) | TF-IDF, Elasticsearch | Catches what embeddings miss (e.g. "Residuals") | Dumb to synonyms — that's why it's hybrid, not solo |
| **FastEmbed (ONNX)** | Turns text into vectors | OpenAI embeddings, sentence-transformers | Loads in ~0.3s on CPU, free, no torch | Slightly weaker than big models on tricky paraphrase |
| **Groq / Ollama / …** | Writes the final answer | OpenAI, Gemini, local llama.cpp | Groq free tier is fast + documented; Ollama is unlimited + private | Free tiers change; local needs ~8GB RAM for good models |

### "What if I don't have X?"

- **No GPU?** You don't need one. Everything here is CPU-first (ONNX embeddings, API LLMs, or small Ollama models like `llama3.2`).
- **No API key?** Use Ollama (host installs once, visitors never see keys) or the anonymous Pollinations fallback.
- **No Ollama?** Set `LLM_PROVIDER` + a free Groq/Gemini/OpenRouter key in `.env` — 2 minutes, no install.
- **Video has no captions?** Nothing to retrieve — YOUBO will tell you. Auto-generated captions count, so most videos work.
- **Huge playlist?** Loading caps at 20 videos / transcript size caps keep memory sane. Load in batches for monster playlists.
- **Slow first load?** That's the embedding model downloading once (~100MB, cached after). Check Diagnostics timings to see which step costs.

### Deep questions, straight answers

- **Why hybrid (vector + keyword)?** Vectors find *meaning* ("songs about heartbreak"); keywords find *exact strings* ("Residuals", "Mosh"). Either alone misses things.
- **Why timestamps?** A RAG answer without sources is a rumor. Every chunk carries its `&t=` link so you can verify in one click.
- **Why chat history rewriting?** Follow-ups like "what about the second one?" are meaningless alone — the history-aware retriever rewrites them into standalone questions first.
- **Is my data private?** With Ollama: yes, 100% local. With cloud LLMs: your questions + retrieved chunks go to that provider (their policy applies). Chat logs stay on this server only.
"""

TARGET_LANGS = ["Hindi", "Spanish", "French", "German", "Portuguese",
                 "Arabic", "Tamil", "Telugu", "Bengali", "Marathi",
                 "Japanese", "Korean", "English"]


def translate_text(llm, text: str, target_lang: str, max_chars: int = 12000) -> str:
    prompt = (
        f"Translate the following markdown content into {target_lang}. "
        "Keep all markdown formatting, headings, bullet points and links intact. "
        "Reply with ONLY the translation.\n\n" + text[:max_chars]
    )
    return llm.invoke(prompt).content


def build_key_moments(docs: list[Document], max_items: int = 8) -> list[dict]:
    """Pick evenly spaced chunks across videos as 'key moments' with links."""
    if not docs:
        return []
    step = max(1, len(docs) // max_items)
    moments = []
    for i in range(0, len(docs), step):
        d = docs[i]
        moments.append({
            "link": d.metadata.get("link", ""),
            "start": d.metadata.get("start", 0),
            "snippet": d.page_content[:120].strip() + "...",
        })
        if len(moments) >= max_items:
            break
    return moments


def notes_to_pdf(notes_md: str) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", size=11)
    for line in notes_md.splitlines():
        safe = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 7, safe)
    return bytes(pdf.output())
