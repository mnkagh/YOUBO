"""YouTube RAG Chatbot — Streamlit UI."""
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import utils

load_dotenv()
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon=":tv:", layout="wide")
st.title("Chat with YouTube Videos")
st.caption("Hybrid retrieval (Qdrant + BM25) with timestamped citations.")

provider, model, api_key = utils.resolve_provider()

try:
    llm = utils.get_llm(provider, api_key, model)
except Exception as e:
    st.error(
        "The AI backend isn't available right now. "
        "(Host: set LLM_PROVIDER + key in .env, or start Ollama.)"
    )
    st.stop()

session_id = st.text_input("Session ID", value="default_session")

for key, default in [("store", {}), ("video_docs", []), ("video_ids", []), ("video_meta", {}), ("notes", None), ("quiz", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


with st.sidebar:
    st.header("Videos")
    raw_urls = st.text_area(
        "YouTube URLs / video IDs (one per line). Playlists expand to the first 10 videos.",
        height=140,
        placeholder="https://www.youtube.com/watch?v=...",
    )
    load_btn = st.button("Load videos")
    compare_mode = st.checkbox("Multi-video comparison mode", value=False)

    with st.expander("Diagnostics (server log)"):
        st.caption(f"Backend: {provider} / {model}")
        st.caption(f"Index: {len(st.session_state.video_ids)} video(s), {len(st.session_state.video_docs)} chunks")
        st.code(utils.read_recent_logs(40), language="text")

    if st.session_state.video_meta:
        st.subheader("Loaded")
        for vid in st.session_state.video_ids:
            meta = st.session_state.video_meta.get(vid, {})
            st.markdown(f"**{meta.get('title', vid)}**")
            duration = meta.get("duration") or 0
            st.caption(f"{meta.get('uploader', '')} · {duration // 60}m {duration % 60}s")

if load_btn and raw_urls:
    video_ids = []
    for line in raw_urls.splitlines()[:50]:
        line = line.strip()
        if not line:
            continue
        if utils.is_playlist_url(line):
            for entry in utils.expand_playlist(line):
                vid = utils.extract_video_id(str(entry)) or str(entry)
                if vid:
                    video_ids.append(vid)
        else:
            vid = utils.extract_video_id(line)
            if vid:
                video_ids.append(vid)
    video_ids = list(dict.fromkeys(video_ids))[: utils.MAX_VIDEOS]
    if not video_ids:
        st.error("No valid video IDs found.")
    else:
        all_docs, meta = [], {}
        with st.status("Loading videos...", expanded=True) as status:
            st.write("Fetching transcripts...")
            for vid in video_ids:
                try:
                    with utils.Timer(f"transcript {vid}"):
                        all_docs.extend(utils.transcript_to_docs(vid, utils.fetch_transcript(vid)))
                    meta[vid] = utils.get_video_metadata(vid)
                except Exception as e:
                    utils.log_error(f"transcript {vid}", e)
                    st.warning(f"Skipping {vid}: {e}")
            if all_docs:
                st.write("Building search index (first run downloads the embedding model)...")
                try:
                    with utils.Timer(f"index {len(video_ids)} video(s)"):
                        st.session_state.retriever_resources = utils.build_hybrid_retriever(video_ids, all_docs)
                    st.session_state.retriever_cache_key = tuple(video_ids)
                    status.update(label="Videos loaded.", state="complete")
                except Exception as e:
                    utils.log_error("index build", e)
                    status.update(label="Index build failed — see Diagnostics.", state="error")
                    st.error(f"Could not build the search index: {e}")
                    all_docs = []
        if not all_docs:
            st.error("No transcripts available. Videos may lack captions.")
        else:
            st.session_state.video_docs = all_docs
            st.session_state.video_ids = video_ids
            st.session_state.video_meta = meta
            st.success(f"Loaded {len(video_ids)} video(s), {len(all_docs)} chunks.")

if st.session_state.video_docs:
    docs = st.session_state.video_docs
    cache_key = tuple(st.session_state.video_ids)
    if st.session_state.get("retriever_cache_key") != cache_key:
        # Fallback path (normally built during Load with progress UI).
        try:
            with st.spinner("Building search index..."):
                st.session_state.retriever_resources = utils.build_hybrid_retriever(
                    st.session_state.video_ids, docs
                )
            st.session_state.retriever_cache_key = cache_key
        except Exception as e:
            utils.log_error("index build", e)
            st.error(f"Could not build the search index: {e}")
            st.stop()
    vectorstore, splits, bm25 = st.session_state.retriever_resources
    base_retriever = utils.HybridRetriever(vectorstore=vectorstore, splits=splits, bm25=bm25, k=4)
    retriever = utils.LinkAttachingRetriever(base=base_retriever)

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
            "Compare and contrast the videos' perspectives, citing each claim with the "
            "matching markdown link. If you don't know, say so.\n\n{context}"
        )
    else:
        system_prompt = (
            "You are an assistant for question-answering over YouTube transcripts. "
            "Answer concisely (max 3 sentences) using the retrieved context. "
            "Cite sources with the chunk's markdown link so the user can jump to that "
            "timestamp. If you don't know, say so.\n\n{context}"
        )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))
    chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    tab_chat, tab_summary, tab_quiz, tab_notes = st.tabs(["Chat", "Summary", "Quiz", "Notes & Export"])

    with tab_chat:
        history = get_session_history(session_id)
        for i, msg in enumerate(history.messages):
            with st.chat_message("user" if i % 2 == 0 else "assistant"):
                st.markdown(msg.content)
        user_input = st.chat_input("Ask about the video(s):")
        if user_input:
            if len(user_input) > 2000:
                st.error("Question too long (max 2000 chars).")
            else:
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.spinner("Thinking..."):
                    try:
                        response = chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}},
                        )
                        with st.chat_message("assistant"):
                            st.markdown(response["answer"])
                    except Exception as e:
                        utils.log_error("chat answer", e)
                        st.error("Sorry, that answer failed. The error was logged — see Diagnostics in the sidebar.")
                st.rerun()

    with tab_summary:
        level = st.radio("Summary length", ["TL;DR", "Short", "Detailed"], horizontal=True)
        if st.button("Generate summary"):
            full_text = "\n".join(d.page_content for d in docs)
            with st.spinner("Summarizing..."):
                try:
                    st.session_state.summary = utils.build_summary(llm, full_text, level)
                except Exception as e:
                    utils.log_error("summary", e)
                    st.error("Summary failed. The error was logged — see Diagnostics in the sidebar.")
        if st.session_state.get("summary"):
            st.markdown(st.session_state.summary)
            st.download_button("Download summary (.md)", st.session_state.summary,
                               file_name="summary.md", mime="text/markdown")

    with tab_quiz:
        n_q = st.slider("Number of questions", 3, 10, 5)
        if st.button("Generate quiz"):
            full_text = "\n".join(d.page_content for d in docs)
            with st.spinner("Creating quiz..."):
                try:
                    st.session_state.quiz = utils.build_quiz(llm, full_text, n=n_q)
                except Exception as e:
                    utils.log_error("quiz", e)
                    st.error("Quiz generation failed. The error was logged — see Diagnostics in the sidebar.")
        if st.session_state.quiz:
            st.markdown(st.session_state.quiz)
            st.download_button("Download quiz (.md)", st.session_state.quiz,
                               file_name="quiz.md", mime="text/markdown")

    with tab_notes:
        if st.button("Generate study notes"):
            full_text = "\n".join(d.page_content for d in docs)
            with st.spinner("Summarizing..."):
                try:
                    st.session_state.notes = utils.build_notes(llm, full_text)
                except Exception as e:
                    utils.log_error("notes", e)
                    st.error("Notes generation failed. The error was logged — see Diagnostics in the sidebar.")
        if st.session_state.notes:
            st.markdown(st.session_state.notes)
            c1, c2 = st.columns(2)
            c1.download_button("Download notes (.md)", st.session_state.notes,
                               file_name="study_notes.md", mime="text/markdown")
            try:
                c2.download_button("Download notes (.pdf)", utils.notes_to_pdf(st.session_state.notes),
                                   file_name="study_notes.pdf", mime="application/pdf")
            except Exception:
                c2.info("Install fpdf2 for PDF export: pip install fpdf2")

        st.subheader("Key moments")
        for m in utils.build_key_moments(docs):
            st.markdown(f"[{int(m['start'])}s]({m['link']}) — {m['snippet']}")

        history = get_session_history(session_id)
        chat_md = "\n\n".join(
            f"**{'You' if i % 2 == 0 else 'Assistant'}:** {m.content}"
            for i, m in enumerate(history.messages)
        )
        if chat_md:
            st.download_button("Download chat history (.md)", chat_md,
                               file_name="chat_history.md", mime="text/markdown")
else:
    st.info("Load at least one video from the sidebar to start chatting.")
