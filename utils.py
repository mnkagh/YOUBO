"""Core logic for the YouTube RAG chatbot: loading, retrieval, generation helpers."""
import re
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

MAX_VIDEOS = 20
MAX_TRANSCRIPT_CHARS = 200_000

PROVIDERS = {
    "Ollama (local, unlimited, no key)": {"model": "llama3.2", "env": "", "needs_key": False},
    "Groq (free tier: 30 rpm / 1k day)": {"model": "gpt-oss-120b", "env": "GROQ_API_KEY", "needs_key": True},
    "Google Gemini (free Flash models)": {"model": "gemini-3-flash-preview", "env": "GEMINI_API_KEY", "needs_key": True},
    "OpenRouter (free models, 50/day)": {"model": "openai/gpt-oss-120b:free", "env": "OPENROUTER_API_KEY", "needs_key": True},
}

PROVIDER_LINKS = {
    "Groq (free tier: 30 rpm / 1k day)": "https://console.groq.com/keys",
    "Google Gemini (free Flash models)": "https://aistudio.google.com/app/apikey",
    "OpenRouter (free models, 50/day)": "https://openrouter.ai/keys",
    "Ollama (local, unlimited, no key)": "https://ollama.com/download",
}


def get_llm(provider: str, api_key: str, model: str | None = None):
    """Build a chat model for the chosen provider. Every option above works on a free tier."""
    if not provider.startswith("Ollama") and not api_key:
        raise ValueError(f"{provider} needs an API key. Get a free one at {PROVIDER_LINKS[provider]}")
    model = model or PROVIDERS[provider]["model"]

    if provider.startswith("Ollama"):
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0.2)

    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.2)

    if provider.startswith("Google Gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)

    if provider.startswith("OpenRouter"):
        return OpenRouterChat(model=model, api_key=api_key, temperature=0.2)

    raise ValueError(f"Unsupported provider: {provider}")


class OpenRouterChat(BaseChatModel):
    """Minimal OpenRouter client (requests only, no tiktoken dependency)."""

    model: str
    api_key: str
    temperature: float = 0.2
    base_url: str = "https://openrouter.ai/api/v1"

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.type, "content": m.content if isinstance(m.content, str) else str(m.content)}
                for m in messages
            ],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=resp.json()["choices"][0]["message"]["content"]))]
        )


def extract_video_id(text: str) -> str | None:
    text = text.strip()[:500]
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([\w-]{11})", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[\w-]{11}", text):
        return text
    return None


def is_playlist_url(text: str) -> bool:
    return "list=" in text and "youtube" in text or text.strip().startswith("https://www.youtube.com/playlist")


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
        }
    except Exception:
        return {"title": video_id, "duration": 0, "uploader": "unknown", "view_count": None, "thumbnail": None}


def fetch_transcript(video_id: str) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        data = YouTubeTranscriptApi().fetch(video_id)
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in data]
    except AttributeError:
        return YouTubeTranscriptApi.get_transcript(video_id)


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
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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


def build_notes(llm, transcript_text: str, max_chars: int = 12000) -> str:
    prompt = (
        "Turn the following YouTube video transcript into clean, structured study notes "
        "in markdown with headings, bullet points, and key takeaways.\n\n"
        + transcript_text[:max_chars]
    )
    return llm.invoke(prompt).content


def build_summary(llm, transcript_text: str, level: str, max_chars: int = 12000) -> str:
    instructions = {
        "TL;DR": "Write a TL;DR summary in at most 3 sentences.",
        "Short": "Write a short summary in one paragraph (~5 sentences).",
        "Detailed": "Write a detailed summary with sections: Overview, Main Points, Conclusion.",
    }
    prompt = f"{instructions.get(level, instructions['Short'])}\n\n{transcript_text[:max_chars]}"
    return llm.invoke(prompt).content


def build_quiz(llm, transcript_text: str, n: int = 5, max_chars: int = 12000) -> str:
    prompt = (
        f"Based on this YouTube transcript, create a quiz with {n} multiple-choice questions. "
        "Each question must have options A-D, the correct answer marked, and a one-line explanation. "
        "Format in markdown.\n\n" + transcript_text[:max_chars]
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
