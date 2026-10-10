"""YOUBO — Streamlit UI."""
import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import utils
import auth

load_dotenv()
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

st.set_page_config(page_title="YOUBO", page_icon=":tv:", layout="wide")

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"] { font-family: 'Poppins', sans-serif; }
footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }
[data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { max-width: 1100px; }
.brandbar { background: linear-gradient(90deg, #FF0000 0%, #7a0d0d 60%, #1a0505 100%);
  border-radius: 16px; padding: 1.1rem 1.4rem; color: #fff !important; margin-bottom: 1rem;
  box-shadow: 0 8px 28px rgba(255,0,0,.25); }
.brandbar h1 { border: none !important; padding: 0 !important; margin: 0 !important; color: #fff !important; font-weight: 700; }
.brandbar p { margin: .2rem 0 0 0 !important; color: #ffd9d9 !important; }
h1 { border-bottom: 4px solid #FF0000; padding-bottom: .3rem; }
.stButton > button { border-radius: 999px; font-weight: 600; }
.stButton > button[kind="primary"], .stButton > button:hover { border-color: #FF0000; }
.stChatMessage { border-radius: 14px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a0505 0%, #0f0f0f 30%); }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stToggle label { color: #f5f5f5 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { background: #ffffff !important; color: #111111 !important; -webkit-text-fill-color: #111111 !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div { background: #ffffff !important; color: #111111 !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] span, [data-testid="stSidebar"] div[data-baseweb="select"] svg { color: #111111 !important; fill: #111111 !important; }
section[data-testid="stSidebar"] img { border-radius: 10px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.stTabs [aria-selected="true"] { color: #FF0000 !important; }
.stMetric { background: #fff5f5; border: 1px solid #ffc9c9; border-radius: 12px; padding: .5rem; }
/* YOUBO-only chrome: hide every Streamlit-branded element */
#MainMenu, footer, .stAppDeployButton, [data-testid="stToolbar"],
[data-testid="stStatusWidget"] { visibility: hidden !important; display: none !important; }
[data-testid="stDecoration"] { background: linear-gradient(90deg, #FF0000, #7a0000) !important; }
div[data-testid="stChatInput"] textarea { background: #ffffff !important; color: #111111 !important; -webkit-text-fill-color: #111111 !important; }
div[data-testid="stChatInput"] textarea::placeholder { color: #777777 !important; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap: .4rem; }
div[data-testid="stRadio"] label { background: rgba(128,128,128,.12); border-radius: 999px; padding: .35rem .9rem; }
div[data-testid="stRadio"] label:has(input:checked) { background: #FF0000 !important; }
div[data-testid="stRadio"] label:has(input:checked) p { color: #ffffff !important; }
.hero { text-align: center; padding: 1.2rem .5rem; }
.hero h2 { margin-bottom: .2rem; }
.step-cards { display: flex; gap: .6rem; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)
# Kill zombie widgets: if the browser serves a cached copy of the page
# (back-forward cache), force a hard reload so inputs are live again.
st.html(
    "<script>window.addEventListener('pageshow', function (e) { "
    "if (e.persisted) { window.location.reload(); } });</script>",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # black by default

DARK_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #0b0b0f !important; }
.block-container, .main, p, li, span, label, h1, h2, h3, h4, .stMarkdown { color: #f1f1f1 !important; }
.stTextInput input, .stTextArea textarea { background: #1c1c22 !important; color: #ffffff !important; border: 1px solid #3a3a44 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { background: #ffffff !important; color: #111111 !important; -webkit-text-fill-color: #111111 !important; border: 1px solid #cccccc !important; }
.stChatMessage { background: #141419 !important; border: 1px solid #26262e !important; }
div[data-testid="stChatInput"] textarea { background: #1c1c22 !important; color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-testid="stChatInput"] textarea::placeholder { color: #999999 !important; }
.stTabs [data-baseweb="tab"] { color: #bbbbbb !important; }
.stTabs [data-baseweb="tab-list"] { background: #0b0b0f !important; }
.stExpander, details { background: #141419 !important; border-color: #26262e !important; }
.stExpander summary, .stExpander p, .stExpander span { color: #f1f1f1 !important; }
.stRadio label, .stCheckbox label, .stSelectbox label { color: #f1f1f1 !important; }
.stButton > button { background: #1c1c22 !important; color: #ffffff !important; border: 1px solid #3a3a44 !important; }
.stButton > button:hover { border-color: #FF0000 !important; color: #ffffff !important; }
.stDownloadButton > button { background: #2a0d0d !important; color: #ffb3b3 !important; border: 1px solid #7a1f1f !important; }
.stMetric { background: #1a1010 !important; border: 1px solid #5c1a1a !important; }
.stMetric label, .stMetric div { color: #ffd7d7 !important; }
code, pre, [data-testid="stCodeBlock"] { background: #141419 !important; }
[data-testid="stCodeBlock"] code { color: #e8e8e8 !important; }
.stSlider label { color: #f1f1f1 !important; }
.stAlert { filter: brightness(.92); }
hr { border-color: #2a2a32 !important; }
</style>
"""
if st.session_state.dark_mode:
    st.markdown(DARK_CSS, unsafe_allow_html=True)

st.markdown('<div class="brandbar"><h1>YOUBO</h1><p>Chat with YouTube videos — answers with timestamped proof.</p></div>',
            unsafe_allow_html=True)


def state_owner() -> str | None:
    """Who to save state under: the logged-in user, or the anonymous guest token."""
    if st.session_state.get("auth_user"):
        return st.session_state.auth_user
    sid = st.session_state.get("guest_sid")
    if st.session_state.get("guest") and sid:
        return f"_guest_{sid}"
    return None


def persist_user_state() -> None:
    owner = state_owner()
    if not owner:
        return
    auth.save_state(owner, {
        "dark_mode": st.session_state.get("dark_mode", True),
        "raw_urls": st.session_state.get("raw_urls_box", ""),
        "lang": st.session_state.get("lang_choice", "Auto (any available)"),
        "compare": st.session_state.get("compare_mode", False),
        "video_ids": st.session_state.video_ids,
        "video_meta": st.session_state.video_meta,
        "video_lang": st.session_state.video_lang,
        "chat_names": st.session_state.chat_names,
        "active_chat": st.session_state.active_chat,
        "transcript_tr": st.session_state.get("transcript_tr"),
        "chats": {cid: auth.history_to_list(h) for cid, h in st.session_state.store.items()},
    })


def apply_saved_state(saved: dict) -> None:
    st.session_state.dark_mode = saved.get("dark_mode", True)
    st.session_state.chat_names = saved.get("chat_names", {})
    st.session_state.active_chat = saved.get("active_chat")
    st.session_state.store = {
        cid: auth.list_to_history(items) for cid, items in saved.get("chats", {}).items()
    }
    st.session_state.raw_urls_box = saved.get("raw_urls", "")
    st.session_state.lang_choice = saved.get("lang", "Auto (any available)")
    st.session_state.compare_mode = saved.get("compare", False)
    st.session_state.restore_videos = saved.get("video_ids", [])
    st.session_state.video_meta = saved.get("video_meta", {})
    st.session_state.video_lang = saved.get("video_lang", "")
    if not st.session_state.active_chat or st.session_state.active_chat not in st.session_state.store:
        new_chat()


if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
    st.session_state.guest = False
    st.session_state.guest_sid = None
    auth.prune_old_guests()
    # Returning guest? The ?s= token in the URL identifies this browser.
    try:
        token = st.query_params.get("s", "")
    except Exception:
        token = ""
    if token and auth.guest_state_exists(token):
        import re as _re
        if _re.fullmatch(r"[0-9a-f]{16}", token):
            st.session_state.guest = True
            st.session_state.guest_sid = token
            st.session_state.restore_pending = "_guest_" + token

if st.session_state.auth_user is None and not st.session_state.guest:
    t_login, t_signup, t_guest = st.tabs(["Login", "Sign up", "Guest"])
    with t_login:
        u = st.text_input("Username", key="li_user")
        p = st.text_input("Password", type="password", key="li_pass")
        if st.button("Login", type="primary"):
            if auth.verify(u, p):
                st.session_state.auth_user = u.strip().lower()
                st.session_state.restore_pending = st.session_state.auth_user
                st.rerun()
            else:
                st.error("Wrong username or password.")
    with t_signup:
        u = st.text_input("Username", key="su_user")
        p = st.text_input("Password", type="password", key="su_pass")
        if st.button("Create account", type="primary"):
            err = auth.signup(u, p)
            if err:
                st.error(err)
            else:
                st.session_state.auth_user = u.strip().lower()
                st.success("Account created — you're logged in.")
                st.rerun()
    with t_guest:
        st.write("Guest mode: everything works and your work survives refresh on this browser. Login to keep it across devices.")
        if st.button("Continue as guest", type="primary"):
            import secrets as _secrets
            st.session_state.guest = True
            st.session_state.guest_sid = _secrets.token_hex(8)
            try:
                st.query_params["s"] = st.session_state.guest_sid
            except Exception:
                pass
            st.rerun()
    st.stop()

if st.session_state.get("restore_pending"):
    owner = st.session_state.restore_pending
    st.session_state.restore_pending = False
    saved = auth.load_state(owner) if isinstance(owner, str) and owner else {}
    if saved:
        apply_saved_state(saved)
        n_chats = len(saved.get("chat_names", {}))
        n_vids = len(saved.get("video_ids", []))
        if n_chats or n_vids:
            who = st.session_state.auth_user or "guest"
            st.session_state.welcome_back = f"Welcome back, {who} — restored {n_chats} chat(s), {n_vids} video(s)."
    st.rerun()

_welcome = st.session_state.pop("welcome_back", None)
if _welcome:
    st.toast(_welcome)

provider, model, api_key = utils.resolve_provider()

# Guest refresh safety: keep the pasted links in the URL so a browser
# refresh restores them even though server-side state is wiped.
try:
    qp_urls = st.query_params.get("urls", "")
    if qp_urls and "raw_urls_box" not in st.session_state:
        st.session_state.raw_urls_box = qp_urls
except Exception:
    pass

try:
    llm = utils.get_llm(provider, api_key, model)
except Exception as e:
    st.error(
        "The AI backend isn't available right now. "
        "(Host: set LLM_PROVIDER + key in .env, or start Ollama.)"
    )
    st.stop()

for key, default in [("store", {}), ("chat_names", {}), ("active_chat", None),
                     ("video_docs", []), ("video_ids", []), ("video_meta", {}),
                     ("video_lang", ""), ("notes", None), ("quiz", None),
                     ("quiz_data", None), ("quiz_done", False),
                     ("transcript_tr", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def new_chat(name: str | None = None) -> str:
    """Create a chat with an auto-generated id. Users never see or type ids."""
    chat_id = uuid.uuid4().hex[:8]
    st.session_state.store[chat_id] = ChatMessageHistory()
    n = len(st.session_state.chat_names) + 1
    st.session_state.chat_names[chat_id] = name or f"Chat {n}"
    st.session_state.active_chat = chat_id
    return chat_id


if st.session_state.active_chat is None:
    new_chat()


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


with st.sidebar:
    dark = st.toggle("Dark mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        persist_user_state()
        st.rerun()
    if st.session_state.auth_user:
        st.caption(f"Logged in as **{st.session_state.auth_user}** — chats auto-save.")
        if st.button("Logout"):
            persist_user_state()
            st.session_state.auth_user = None
            st.session_state.guest = False
            for k in ("store", "chat_names", "active_chat", "video_docs", "video_ids",
                      "video_meta", "video_lang", "notes", "quiz_data", "raw_urls_box"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.caption("Guest mode — auto-saved in this browser, even on refresh.")
    st.header("Videos")
    raw_urls = st.text_area(
        "YouTube URLs / video IDs (one per line). Playlists expand to the first 10 videos.",
        height=140,
        placeholder="https://www.youtube.com/watch?v=...",
        key="raw_urls_box",
    )
    load_btn = st.button("Load videos", type="primary")
    lang_choice = st.selectbox("Transcript language", list(utils.LANGUAGES), index=0, key="lang_choice")
    compare_mode = st.checkbox("Multi-video comparison mode", value=False, key="compare_mode")

    st.subheader("Chats")
    if st.button("+ New chat"):
        new_chat()
        st.rerun()
    chat_ids = list(st.session_state.chat_names)
    if chat_ids:
        current = st.session_state.active_chat
        picked = st.radio(
            "Switch chat",
            chat_ids,
            index=chat_ids.index(current) if current in chat_ids else 0,
            format_func=lambda c: f"{st.session_state.chat_names[c]} ({len(st.session_state.store.get(c, ChatMessageHistory()).messages) // 2} Q)",
            label_visibility="collapsed",
        )
        if picked != st.session_state.active_chat:
            st.session_state.active_chat = picked
            st.rerun()
        c1, c2 = st.columns(2)
        new_name = c1.text_input("Rename", value=st.session_state.chat_names[st.session_state.active_chat])
        if new_name != st.session_state.chat_names[st.session_state.active_chat]:
            st.session_state.chat_names[st.session_state.active_chat] = new_name[:60]
            st.rerun()
        if c2.button("Delete") and len(chat_ids) > 1:
            gone = st.session_state.active_chat
            del st.session_state.store[gone]
            del st.session_state.chat_names[gone]
            st.session_state.active_chat = list(st.session_state.chat_names)[0]
            st.rerun()

    with st.expander("Diagnostics (server log)"):
        st.caption(f"Backend: {provider} / {model}")
        st.caption(f"Index: {len(st.session_state.video_ids)} video(s), {len(st.session_state.video_docs)} chunks")
        st.code(utils.read_recent_logs(40), language="text")

    if st.session_state.video_meta:
        st.subheader("Loaded")
        for vid in st.session_state.video_ids:
            meta = st.session_state.video_meta.get(vid, {})
            if meta.get("thumbnail"):
                st.image(meta["thumbnail"], use_container_width=True)
            st.markdown(f"**{meta.get('title', vid)}**")
            duration = meta.get("duration") or 0
            st.caption(f"{meta.get('uploader', '')} · {duration // 60}m {duration % 60}s")


def load_video_ids(video_ids: list, lang_code: str, label: str = "Loading videos...") -> bool:
    """Fetch transcripts + build index. Returns True on success."""
    all_docs, meta, langs = [], {}, []
    with st.status(label, expanded=True) as status:
        st.write("Fetching transcripts...")
        for vid in video_ids:
            try:
                with utils.Timer(f"transcript {vid}"):
                    snippets, lang_name = utils.fetch_transcript(vid, lang_code)
                    all_docs.extend(utils.transcript_to_docs(vid, snippets))
                langs.append(lang_name)
                meta[vid] = utils.get_video_metadata(vid)
            except Exception as e:
                utils.log_error(f"transcript {vid}", e)
                st.write(f"No captions for {vid} — transcribing audio locally (slower, one-time)...")
                try:
                    snippets, lang_name = utils.transcribe_audio(vid)
                    all_docs.extend(utils.transcript_to_docs(vid, snippets))
                    langs.append(lang_name)
                    meta[vid] = utils.get_video_metadata(vid)
                except Exception as e2:
                    utils.log_error(f"audio fallback {vid}", e2)
                    st.warning(f"Skipping {vid}: no captions and audio transcription failed ({e2})")
        if all_docs:
            st.write("Building search index...")
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
        return False
    st.session_state.video_docs = all_docs
    st.session_state.video_ids = video_ids
    st.session_state.video_meta = meta
    st.session_state.video_lang = ", ".join(sorted(set(langs))) or "unknown"
    st.success(f"Loaded {len(video_ids)} video(s), {len(all_docs)} chunks (captions: {st.session_state.video_lang}).")
    persist_user_state()
    return True


triggered = st.session_state.pop("trigger_load", False)
if load_btn or triggered:
    if not (raw_urls or "").strip():
        st.warning("Paste a YouTube link first — the box is empty on the server. "
                   "If you can see a link, click inside the box, press space then backspace, and try Load again.")
    else:
        try:
            st.query_params["urls"] = raw_urls
        except Exception:
            pass
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
            load_video_ids(video_ids, utils.LANGUAGES[lang_choice])

if st.session_state.pop("restore_videos", None) and not st.session_state.video_docs:
    saved_ids = st.session_state.get("video_ids", [])
    if saved_ids:
        load_video_ids(saved_ids, utils.LANGUAGES.get(st.session_state.get("lang_choice", "Auto (any available)"), "auto"),
                       label="Restoring your videos...")

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
    video_titles = [st.session_state.video_meta.get(v, {}).get("title", v)
                    for v in st.session_state.video_ids]
    content_types = sorted({utils.content_type_of(st.session_state.video_meta.get(v, {}))
                            for v in st.session_state.video_ids})
    scope = (
        f"You know ONLY these loaded video(s): {'; '.join(video_titles)}. "
        f"Content type(s): {', '.join(content_types)}. "
        "If asked about videos, playlists, or links NOT in this list, say clearly "
        "that only the loaded video(s) are available and name them. Never invent "
        "video counts, links, or playlist contents."
    )
    cite_rules = (
        "Cite every factual claim with the exact [Source](url) markdown link found "
        "in the retrieved context. Never invent timestamps, never use any other "
        "citation format (no brackets like 【】, no footnotes)."
    )
    if compare_mode:
        system_prompt = (
            "You compare what different videos say. "
            f"{scope} Compare and contrast perspectives across the loaded videos. "
            f"{cite_rules} If the answer isn't in the context, say so.\n\n{{context}}"
        )
    else:
        system_prompt = (
            "You answer questions directly about YouTube video content. "
            f"{scope} Answer helpfully and concretely from the retrieved context — "
            "no hedging, no 'as an AI'. "
            f"{cite_rules} If the answer isn't in the context, say what you do know "
            "from the videos instead of refusing.\n\n{context}"
        )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))
    chain_key = (st.session_state.retriever_cache_key, compare_mode, provider, model)
    if st.session_state.get("chain_key") != chain_key:
        st.session_state.chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        st.session_state.chain_key = chain_key
    chain = st.session_state.chain

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Videos", len(st.session_state.video_ids))
    m2.metric("Chunks", len(docs))
    m3.metric("Captions", (st.session_state.video_lang or "unknown")[:14])
    m4.metric("AI", provider.split(" (")[0][:14])

    # Persistent section bar: st.tabs resets to the first tab on every rerun,
    # a radio keeps the user where they were.
    SECTIONS = {"Chat": "Chat", "Summary": "Summary", "Quiz": "Quiz",
                "Notes & Export": "Notes", "Translate": "Translate", "Guide": "Guide"}
    if st.session_state.get("section") not in SECTIONS:
        st.session_state.section = "Chat"
    section = st.radio("Section", list(SECTIONS), index=list(SECTIONS).index(st.session_state.section),
                       horizontal=True, label_visibility="collapsed", key="tabbar")
    st.session_state.section = section

    @st.fragment
    def quiz_fragment():
        """Answer clicks rerun ONLY this fragment — instant, no full-page reload."""
        if not st.session_state.get("quiz_data"):
            st.info("Generate a quiz to start answering.")
            return
        for i, q in enumerate(st.session_state.quiz_data):
            with st.container(border=True):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                st.radio(f"q{i}", q["options"], index=None, key=f"quiz_a_{i}", label_visibility="collapsed")
            if st.session_state.quiz_done:
                picked = st.session_state.get(f"quiz_a_{i}")
                correct = picked == q["options"][q["answer"]]
                st.markdown("Correct!" if correct else f"Wrong — answer: **{q['options'][q['answer']]}**")
                st.caption(q["explanation"])
        c1, c2 = st.columns(2)
        if c1.button("Check answers"):
            st.session_state.quiz_done = True
            st.rerun(scope="fragment")
        if st.session_state.quiz_done:
            score = sum(
                1 for i, q in enumerate(st.session_state.quiz_data)
                if st.session_state.get(f"quiz_a_{i}") == q["options"][q["answer"]]
            )
            c2.metric("Score", f"{score}/{len(st.session_state.quiz_data)}")

    if section == "Chat":
        history = get_session_history(st.session_state.active_chat)
        for i, msg in enumerate(history.messages):
            with st.chat_message("user" if i % 2 == 0 else "assistant"):
                st.markdown(msg.content)
        if not history.messages:
            st.caption("Try one:")
            chip_cols = st.columns(3)
            for c, suggestion in zip(chip_cols, ["Summarize this video", "Key takeaways", "Explain it simply"]):
                if c.button(suggestion, use_container_width=True, key=f"chip_{suggestion}"):
                    st.session_state.preset_q = suggestion
                    st.rerun()
        user_input = st.chat_input("Ask about the video(s):")
        question = st.session_state.pop("preset_q", None) or user_input
        if question:
            if len(question) > 2000:
                st.error("Question too long (max 2000 chars).")
            else:
                with st.chat_message("user"):
                    st.markdown(question)
                try:
                    with st.chat_message("assistant"):
                        ph = st.empty()
                        buf: list = []

                        def _answer_stream():
                            for chunk in chain.stream(
                                {"input": question},
                                config={"configurable": {"session_id": st.session_state.active_chat}},
                            ):
                                text = chunk.get("answer") if isinstance(chunk, dict) else None
                                if isinstance(text, str) and text:
                                    buf.append(text)
                                    yield text
                        for _piece in _answer_stream():
                            ph.markdown("".join(buf) + "▌")
                        ph.markdown("".join(buf))
                    persist_user_state()
                except Exception as e:
                    utils.log_error("chat answer", e)
                    st.error("Sorry, that answer failed. The error was logged — see Diagnostics in the sidebar.")

    main_type = content_types[0] if content_types else "general"

    if section == "Summary":
        level = st.radio("Summary length", ["TL;DR", "Short", "Detailed"], horizontal=True)
        if st.button("Generate summary"):
            full_text = "\n".join(d.page_content for d in docs)
            prompt = utils.summary_prompt(full_text, level, main_type)
            try:
                ph = st.empty()
                buf: list = []
                for tok in utils.stream_answer(llm, prompt):
                    buf.append(tok)
                    ph.markdown("".join(buf) + "▌")
                ph.markdown("".join(buf))
                st.session_state.summary = "".join(buf)
                persist_user_state()
            except Exception as e:
                utils.log_error("summary", e)
                st.error("Summary failed. The error was logged — see Diagnostics in the sidebar.")
        if st.session_state.get("summary"):
            st.markdown(st.session_state.summary)
            st.download_button("Download summary (.md)", st.session_state.summary,
                               file_name="summary.md", mime="text/markdown")

    if section == "Quiz":
        st.caption(f"Quiz adapts to this content: {main_type}. Answer, then check your score with explanations.")
        n_q = st.slider("Number of questions", 3, 10, 5)
        if st.button("Generate quiz"):
            full_text = "\n".join(d.page_content for d in docs)
            with st.spinner("Creating quiz..."):
                try:
                    st.session_state.quiz_data = utils.build_quiz_data(llm, full_text, main_type, n=n_q)
                    st.session_state.quiz_done = False
                    for k in list(st.session_state.keys()):
                        if k.startswith("quiz_a_"):
                            del st.session_state[k]
                except Exception as e:
                    utils.log_error("quiz", e)
                    st.error("Quiz generation failed. The error was logged — see Diagnostics in the sidebar.")
        quiz_fragment()

    if section == "Notes & Export":
        if st.button("Generate study notes"):
            full_text = "\n".join(d.page_content for d in docs)
            prompt = utils.notes_prompt(full_text, main_type)
            try:
                ph = st.empty()
                buf: list = []
                for tok in utils.stream_answer(llm, prompt):
                    buf.append(tok)
                    ph.markdown("".join(buf) + "▌")
                ph.markdown("".join(buf))
                st.session_state.notes = "".join(buf)
                persist_user_state()
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
        moments = utils.build_key_moments(docs)
        for r in range(0, len(moments), 2):
            cols = st.columns(2)
            for c, m in zip(cols, moments[r:r + 2]):
                with c:
                    with st.container(border=True):
                        mins, secs = divmod(int(m["start"]), 60)
                        st.link_button(f"Play from {mins}:{secs:02d}", m["link"], use_container_width=True)
                        st.caption(m["snippet"])

        history = get_session_history(st.session_state.active_chat)
        chat_md = "\n\n".join(
            f"**{'You' if i % 2 == 0 else 'Assistant'}:** {m.content}"
            for i, m in enumerate(history.messages)
        )
        if chat_md:
            st.download_button("Download chat history (.md)", chat_md,
                               file_name="chat_history.md", mime="text/markdown")

    if section == "Translate":
        st.caption("Get the video in your language: full timestamped transcript, summary, or notes.")
        target = st.selectbox("Target language", utils.TARGET_LANGS, index=0)
        n_calls = (len(docs) + 3) // 4
        if st.button(f"Translate full transcript ({n_calls} short calls)", type="primary"):
            try:
                bar = st.progress(0.0, text="Translating...")
                blocks = utils.translate_transcript(
                    llm, docs, target, batch_chunks=4, progress_cb=lambda f: bar.progress(f))
                bar.empty()
                st.session_state.transcript_tr = {"lang": target, "blocks": blocks}
                persist_user_state()
            except Exception as e:
                utils.log_error("translate transcript", e)
                st.error("Translation failed. The error was logged — see Diagnostics.")
        tr = st.session_state.get("transcript_tr")
        if tr and tr.get("lang") == target:
            st.subheader(f"Full transcript ({target})")
            for b in tr["blocks"]:
                st.markdown(f"**[{int(b['start'])}s]({b['link']})**")
                st.markdown(b["text"])
            st.download_button(
                "Download translated transcript (.md)",
                utils.transcript_blocks_to_md(tr["blocks"], target),
                file_name=f"transcript_{target}.md", mime="text/markdown")
        elif tr:
            st.info(f"A {tr['lang']} translation exists — pick {tr['lang']} above to view it, or re-translate.")
        st.divider()

        st.subheader("Dub this video")
        st.caption("Same picture, new language: AI voiceover mixed onto the original video. "
                   f"Capped at {utils.MAX_DUB_SECONDS // 60} min per video.")
        if st.button("Create dubbed video", type="primary"):
            try:
                bar = st.progress(0.0, text="Starting...")
                mp4 = utils.dub_video(
                    st.session_state.video_ids[0], llm, target,
                    utils.LANGUAGES.get(st.session_state.get("lang_choice", "Auto (any available)"), "auto"),
                    progress_cb=lambda f, msg="": bar.progress(min(f, 1.0), text=msg))
                bar.empty()
                st.session_state.dubbed_mp4 = mp4
                st.session_state.dubbed_lang = target
                st.success("Dubbed video ready — preview below, download to keep it.")
            except Exception as e:
                utils.log_error("dub video", e)
                st.error(f"Dubbing failed: {e}")
        if st.session_state.get("dubbed_mp4") and st.session_state.get("dubbed_lang") == target:
            st.video(st.session_state.dubbed_mp4, format="video/mp4")
            st.download_button("Download dubbed video (.mp4)", st.session_state.dubbed_mp4,
                               file_name=f"dubbed_{target}.mp4", mime="video/mp4")
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("Translate summary", disabled=not st.session_state.get("summary")):
            try:
                ph = st.empty()
                buf: list = []
                for tok in utils.stream_answer(llm, utils.translate_prompt(st.session_state.summary, target)):
                    buf.append(tok)
                    ph.markdown("".join(buf) + "▌")
                ph.markdown("".join(buf))
                st.session_state.summary_tr = "".join(buf)
                persist_user_state()
            except Exception as e:
                utils.log_error("translate summary", e)
                st.error("Translation failed. The error was logged — see Diagnostics.")
        if c2.button("Translate notes", disabled=not st.session_state.get("notes")):
            try:
                ph = st.empty()
                buf: list = []
                for tok in utils.stream_answer(llm, utils.translate_prompt(st.session_state.notes, target)):
                    buf.append(tok)
                    ph.markdown("".join(buf) + "▌")
                ph.markdown("".join(buf))
                st.session_state.notes_tr = "".join(buf)
                persist_user_state()
            except Exception as e:
                utils.log_error("translate notes", e)
                st.error("Translation failed. The error was logged — see Diagnostics.")
        if not st.session_state.get("summary") and not st.session_state.get("notes"):
            st.info("Generate a summary or notes first.")
        if st.session_state.get("summary_tr"):
            st.subheader(f"Summary ({target})")
            st.markdown(st.session_state.summary_tr)
            st.download_button("Download translation (.md)", st.session_state.summary_tr,
                               file_name=f"summary_{target}.md", mime="text/markdown")
        if st.session_state.get("notes_tr"):
            st.subheader(f"Notes ({target})")
            st.markdown(st.session_state.notes_tr)
            st.download_button("Download translation (.md)", st.session_state.notes_tr,
                               file_name=f"notes_{target}.md", mime="text/markdown")

    if section == "Guide":
        st.markdown(utils.GUIDE_MD)

    persist_user_state()
else:
    st.markdown('<div class="hero"><h2>Turn any YouTube video into a conversation</h2>'
                '<p>Paste a link — ask questions, get timestamped answers, quizzes, notes, translations, even a dubbed video.</p></div>',
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown("**1. Paste**\n\nVideo, playlist, or ID — any language, captions optional.")
    s2.markdown("**2. Ask**\n\nChat with citations, summaries, quizzes, key moments.")
    s3.markdown("**3. Keep**\n\nExport notes, translated transcripts, dubbed MP4s.")
    st.write("")
    if st.button("Try a sample video", type="primary"):
        st.session_state.raw_urls_box = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        st.session_state.trigger_load = True
        st.rerun()
