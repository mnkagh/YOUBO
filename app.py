## Conversational RAG Chatbot for YouTube Videos (timestamps + chat history)
import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

GROQ_MODEL = "llama-3.3-70b-versatile"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## ---------- helpers ----------

def extract_video_id(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([\w-]{11})", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[\w-]{11}", text):
        return text
    return None


def expand_playlist(url: str, limit: int = 10) -> list[str]:
    try:
        import yt_dlp
    except ImportError:
        st.warning("Install yt-dlp to expand playlists: pip install yt-dlp")
        return []
    try:
        with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        entries = info.get("entries", [])[:limit]
        return [e.get("url") or e.get("id") for e in entries if e]
    except Exception as e:
        st.warning(f"Could not expand playlist: {e}")
        return []


def fetch_transcript(video_id: str) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        data = YouTubeTranscriptApi().fetch(video_id)
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in data]
    except AttributeError:
        return YouTubeTranscriptApi.get_transcript(video_id)


def transcript_to_docs(video_id: str, snippets: list[dict]) -> list[Document]:
    chunks = []
    current = []
    size = 0
    start = snippets[0]["start"] if snippets else 0
    for s in snippets:
        current.append(s["text"])
        size += len(s["text"])
        if size >= 1200:
            text = " ".join(current)
            chunks.append(Document(
                page_content=text,
                metadata={
                    "video_id": video_id,
                    "start": start,
                    "link": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
                },
            ))
            current = []
            size = 0
            start = s["start"] + s.get("duration", 0)
    if current:
        chunks.append(Document(
            page_content=" ".join(current),
            metadata={
                "video_id": video_id,
                "start": start,
                "link": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
            },
        ))
    return chunks


class HybridRetriever:
    """Vector (Qdrant) + BM25 keyword retrieval, merged and deduplicated."""

    def __init__(self, docs: list[Document], embedding):
        self.docs = docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = []
        for d in docs:
            splits.extend(splitter.split_documents([d]))
        if not splits:
            splits = docs
        client = QdrantClient(":memory:")
        self.vectorstore = QdrantVectorStore.from_documents(
            splits, embedding, client=client, collection_name="yt_chat"
        )
        self.splits = splits
        self.bm25 = BM25Okapi([d.page_content.lower().split() for d in splits])

    def as_retriever(self, k: int = 4) -> BaseRetriever:
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        parent = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
                vec = vector_retriever.invoke(query)
                scores = parent.bm25.get_scores(query.lower().split())
                top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
                kw = [parent.splits[i] for i in top if scores[i] > 0]
                seen, merged = set(), []
                for d in vec + kw:
                    key = d.page_content[:80]
                    if key not in seen:
                        seen.add(key)
                        merged.append(d)
                    if len(merged) >= k:
                        break
                return merged

        return _Retriever()


def build_notes(llm, transcript_text: str, max_chars: int = 12000) -> str:
    prompt = (
        "Turn the following YouTube video transcript into clean, structured study notes "
        "in markdown with headings, bullet points, and key takeaways.\n\n"
        + transcript_text[:max_chars]
    )
    return llm.invoke(prompt).content


## ---------- UI ----------

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon=":tv:", layout="wide")
st.title("Chat with YouTube Videos")
st.write("Paste a YouTube link (or playlist), ask questions, get answers with timestamped citations.")

api_key = st.text_input("Enter your Groq API key:", type="password", value=os.getenv("GROQ_API_KEY", ""))
session_id = st.text_input("Session ID", value="default_session")

if not api_key:
    st.warning("Please enter your Groq API key. Get one free at https://console.groq.com")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name=GROQ_MODEL)

if "store" not in st.session_state:
    st.session_state.store = {}
if "video_docs" not in st.session_state:
    st.session_state.video_docs = []
if "video_ids" not in st.session_state:
    st.session_state.video_ids = []

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

with st.sidebar:
    st.header("Videos")
    raw_urls = st.text_area(
        "YouTube URLs / video IDs (one per line). Playlists auto-expand to first 10 videos.",
        height=150,
        placeholder="https://www.youtube.com/watch?v=...\nhttps://youtube.com/playlist?list=...",
    )
    process = st.button("Load videos")
    compare_mode = st.checkbox("Multi-video comparison mode", value=False)

if process and raw_urls:
    video_ids = []
    for line in raw_urls.splitlines():
        line = line.strip()
        if not line:
            continue
        if "list=" in line:
            for vid in expand_playlist(line):
                vid = extract_video_id(str(vid)) or str(vid)
                if vid:
                    video_ids.append(vid)
        else:
            vid = extract_video_id(line)
            if vid:
                video_ids.append(vid)
    video_ids = list(dict.fromkeys(video_ids))
    if not video_ids:
        st.error("No valid video IDs found.")
    else:
        all_docs = []
        with st.spinner("Fetching transcripts..."):
            for vid in video_ids:
                try:
                    snippets = fetch_transcript(vid)
                    all_docs.extend(transcript_to_docs(vid, snippets))
                except Exception as e:
                    st.warning(f"Could not get transcript for {vid}: {e}")
        if not all_docs:
            st.error("No transcripts available. Videos may lack captions.")
        else:
            st.session_state.video_docs = all_docs
            st.session_state.video_ids = video_ids
            st.success(f"Loaded {len(video_ids)} video(s), {len(all_docs)} chunks.")

if st.session_state.video_docs:
    docs = st.session_state.video_docs
    hybrid = HybridRetriever(docs, get_embeddings())
    retriever = hybrid.as_retriever(k=4)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference "
                    "context in the chat history, formulate a standalone question. Do NOT answer it, "
                    "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    if compare_mode:
        system_prompt = (
            "You are an assistant that compares what different videos say. "
            "Use the retrieved context (each chunk links to its source video/timestamp). "
            "Compare and contrast the videos' perspectives, and cite each claim with the "
            "matching markdown link from the context. If you don't know, say so.\n\n{context}"
        )
    else:
        system_prompt = (
            "You are an assistant for question-answering over YouTube transcripts. "
            "Use the retrieved context to answer concisely (max 3 sentences). "
            "Cite sources by including the chunk's markdown link so the user can jump "
            "to that timestamp. If you don't know, say so.\n\n{context}"
        )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    def attach_links(docs_in):
        out = []
        for d in docs_in:
            link = d.metadata.get("link")
            label = link or ""
            out.append(Document(
                page_content=f"{d.page_content}\n[Source]({label})" if label else d.page_content,
                metadata=d.metadata,
            ))
        return out

    # wrap retriever so context includes source links
    base_retriever = retriever
    class _LinkRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
            return attach_links(base_retriever.invoke(query))

    history_aware_retriever = create_history_aware_retriever(
        llm, _LinkRetriever(), contextualize_q_prompt
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    st.subheader("Chat")
    history = get_session_history(session_id)
    for i, msg in enumerate(history.messages):
        with st.chat_message("user" if i % 2 == 0 else "assistant"):
            st.markdown(msg.content)

    user_input = st.chat_input("Ask about the video(s):")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Thinking..."):
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
            except Exception as e:
                st.error(f"Error: {e}")
        st.rerun()

    with st.expander("Generate study notes / export"):
        if st.button("Generate study notes (markdown)"):
            full_text = "\n".join(d.page_content for d in docs)
            with st.spinner("Summarizing..."):
                try:
                    notes = build_notes(llm, full_text)
                    st.session_state.notes = notes
                except Exception as e:
                    st.error(f"Error: {e}")
        if "notes" in st.session_state:
            st.markdown(st.session_state.notes)
            st.download_button(
                "Download notes (.md)",
                st.session_state.notes,
                file_name="youtube_study_notes.md",
                mime="text/markdown",
            )
        chat_md = "\n\n".join(
            f"**{'You' if i % 2 == 0 else 'Assistant'}:** {m.content}"
            for i, m in enumerate(history.messages)
        )
        if chat_md:
            st.download_button(
                "Download chat history (.md)",
                chat_md,
                file_name="chat_history.md",
                mime="text/markdown",
            )
else:
    st.info("Load at least one video from the sidebar to start chatting.")
