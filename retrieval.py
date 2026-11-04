"""Chunking the transcript and hybrid retrieval (vectors + BM25)."""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from providers import get_embeddings

MAX_TRANSCRIPT_CHARS = 200_000


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
    """Vector (Qdrant) + BM25 resources over the transcript chunks."""
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


def build_key_moments(docs: list[Document], max_items: int = 8) -> list[dict]:
    """Evenly spaced chunks across the videos, with timestamp links."""
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
