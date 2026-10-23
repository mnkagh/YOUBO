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

st.set_page_config(page_title="YOUBO", page_icon="▶", layout="wide",
                   initial_sidebar_state="expanded")


# Centralized theme tokens + component styles.
# Variables are defined per-theme below; THEME_CSS references them so both
# themes share one component stylesheet.

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"] { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }
footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; visibility: hidden !important; }
[data-testid="stElementToolbar"], a.headerlink { display: none !important; }
.block-container { max-width: 1440px; padding-top: .6rem; padding-bottom: 2rem; }

/* ---- shared structural + token-driven rules ---- */
body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, .main, .block-container { background: var(--yb-bg, #111111) !important; }
.block-container, .main, p, li, span, label, h1, h2, h3, h4, .stMarkdown { color: var(--yb-text, #F5F5F5) !important; }

.stButton > button { border-radius: 8px; font-weight: 600; transition: all .15s ease;
  background: var(--yb-surface, #181818) !important; color: var(--yb-text, #F5F5F5) !important;
  border: 1px solid var(--yb-border, #333333) !important; }
.stButton > button[kind="primary"] { background: var(--yb-primary-btn-bg, #FF4545) !important;
  color: var(--yb-primary-btn-text, #ffffff) !important; border: none !important; }
.stButton > button[kind="primary"]:hover { background: var(--yb-accent-hover, #ff5a5a) !important; }
.stButton > button:hover { border-color: var(--yb-accent, #FF4545) !important; }

.stTextInput > div > div > input, .stTextArea textarea, [data-baseweb="input"] input {
  background: var(--yb-input-bg, #222222) !important; color: var(--yb-input-text, #F5F5F5) !important;
  border: 1px solid var(--yb-border, #333333) !important; border-radius: 8px !important; }
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
  border-color: var(--yb-focus, #FF4545) !important; box-shadow: 0 0 0 1px var(--yb-focus, #FF4545) !important; }

.stChatMessage { background: var(--yb-surface, #181818) !important;
  border: 1px solid var(--yb-border, #333333) !important; border-radius: 12px !important;
  padding: .8rem 1rem !important; }
[data-testid="stChatInput"] textarea { background: var(--yb-input-bg, #222222) !important;
  color: var(--yb-input-text, #F5F5F5) !important; border: 1px solid var(--yb-border, #333333) !important;
  border-radius: 10px !important; -webkit-text-fill-color: var(--yb-input-text, #F5F5F5) !important; }
[data-testid="stChatInput"] button { color: var(--yb-accent, #FF4545) !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--yb-surface-tablist, #181818) !important;
  border-radius: 10px; padding: .2rem; gap: .2rem; border: 1px solid var(--yb-border, #333333); }
.stTabs [data-baseweb="tab"] { color: var(--yb-muted, #A3A3A3) !important; font-weight: 600;
  border-radius: 8px; padding: .45rem .95rem; }
.stTabs [aria-selected="true"] { color: var(--yb-selected-tab-text, #F5F5F5) !important;
  background: var(--yb-selected-tab-bg, #222222) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

.stExpander { background: var(--yb-surface, #181818) !important;
  border: 1px solid var(--yb-border, #333333) !important; border-radius: 10px !important; }
.stExpander summary { color: var(--yb-text, #F5F5F5) !important; font-weight: 600; }
.stExpander summary span, .stExpander p, .stExpander span { color: var(--yb-text, #F5F5F5) !important; }

.stMetric { background: var(--yb-surface, #181818) !important;
  border: 1px solid var(--yb-border, #333333) !important; border-radius: 10px; padding: .6rem .9rem; }
.stMetric label { color: var(--yb-muted, #A3A3A3) !important; font-size: .78rem !important; }
.stMetric [data-testid="stMetricValue"] { color: var(--yb-text, #F5F5F5) !important; }
.stMetric div { color: var(--yb-text, #F5F5F5) !important; }

code, pre, [data-testid="stCodeBlock"] { background: var(--yb-surface-code, #181818) !important;
  border: 1px solid var(--yb-border, #333333); border-radius: 8px; }
[data-testid="stCodeBlock"] code { color: var(--yb-text, #F5F5F5) !important; }

.stAlert { border-radius: 10px; }
.stSelectbox [data-baseweb="select"] > div { background: var(--yb-input-bg, #222222) !important;
  border-color: var(--yb-border, #333333) !important; color: var(--yb-input-text, #F5F5F5) !important; }
.stRadio label, .stCheckbox label, .stSelectbox label { color: var(--yb-text, #F5F5F5) !important; }
.stSlider label { color: var(--yb-muted, #A3A3A3) !important; }
hr { border-color: var(--yb-border, #333333) !important; }

[data-testid="stSidebar"] { background: var(--yb-surface-sidebar, #181818) !important;
  border-right: 1px solid var(--yb-border, #333333); }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stRadio label { color: var(--yb-text, #F5F5F5) !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
  background: var(--yb-input-bg, #222222) !important; color: var(--yb-input-text, #F5F5F5) !important;
  -webkit-text-fill-color: var(--yb-input-text, #F5F5F5) !important; border: 1px solid var(--yb-border, #333333) !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: var(--yb-input-bg, #222222) !important; color: var(--yb-input-text, #F5F5F5) !important;
  border-color: var(--yb-border, #333333) !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] svg { color: var(--yb-input-text, #F5F5F5) !important;
  fill: var(--yb-input-text, #F5F5F5) !important; }

/* ---- YOUBO custom components (token-driven) ---- */
.yb-header { display: flex; align-items: center; gap: 1.1rem; padding: .35rem 0 .85rem;
  border-bottom: 1px solid var(--yb-border, #333333); margin-bottom: 1rem; }
.yb-hdr-left { display: flex; align-items: center; gap: .9rem; }
.yb-hdr-right { display: flex; align-items: center; gap: .6rem; margin-left: auto; }
.yb-logo { font-weight: 700; font-size: 1.15rem; letter-spacing: .08em;
  color: var(--yb-text, #F5F5F5); display: flex; align-items: center; gap: .4rem; }
.yb-logo span { color: var(--yb-accent, #FF4545); }
.yb-nav { color: var(--yb-muted, #A3A3A3); font-size: .88rem; font-weight: 500; }
.yb-hero { text-align: center; padding: 5rem 1rem 3rem; }
.yb-hero h1 { font-size: clamp(2.2rem, 5vw, 3.4rem); font-weight: 700; letter-spacing: -.02em;
  margin: 0 0 .6rem; border: none !important; color: var(--yb-text, #F5F5F5) !important; }
.yb-hero p { color: var(--yb-muted, #A3A3A3); font-size: 1.05rem; max-width: 560px; margin: 0 auto 2rem; }
.yb-chip { display: inline-block; padding: .35rem .85rem; margin: .25rem; border-radius: 999px;
  background: var(--yb-surface, #181818); border: 1px solid var(--yb-border, #333333);
  color: var(--yb-text, #F5F5F5); font-size: .82rem; cursor: pointer; transition: all .15s; }
.yb-chip:hover { border-color: var(--yb-accent, #FF4545); color: var(--yb-accent, #FF4545); }
.yb-card { background: var(--yb-surface, #181818); border: 1px solid var(--yb-border, #333333);
  border-radius: 12px; padding: 1.1rem 1.25rem; }
.yb-muted { color: var(--yb-muted, #A3A3A3) !important; font-size: .88rem; }
.yb-meta { display: flex; gap: .6rem; align-items: center; color: var(--yb-muted, #A3A3A3); font-size: .82rem; }
/* citation badge + download button: shared constants in BOTH themes */
.yb-cite { display: inline-block; padding: .05rem .45rem; margin: 0 .15rem;
  background: var(--yb-cite-bg, #2a1414); border: 1px solid var(--yb-cite-border, #5a2020);
  color: var(--yb-cite-text, #ff8a8a); border-radius: 6px; font-size: .76rem; font-weight: 600; }
.stDownloadButton > button { background: var(--yb-download-bg, #2a1414) !important;
  color: var(--yb-download-text, #ffb3b3) !important; border: 1px solid var(--yb-download-border, #5a2020) !important; }
.yb-section-h { font-size: 1.05rem; font-weight: 600; margin: 1rem 0 .5rem;
  color: var(--yb-text, #F5F5F5); }
.yb-empty { text-align: center; padding: 3rem 1rem; color: var(--yb-muted, #A3A3A3); }
.yb-empty h3 { color: var(--yb-text, #F5F5F5); margin-bottom: .4rem; }
.yb-processing { background: var(--yb-surface, #181818); border: 1px solid var(--yb-border, #333333);
  border-radius: 10px; padding: .7rem .9rem; margin: .5rem 0; }
.yb-processing .step { color: var(--yb-muted, #A3A3A3); font-size: .85rem; padding: .15rem 0; }
.yb-processing .step.done { color: var(--yb-success, #22c55e); }
.yb-processing .step.active { color: var(--yb-accent, #FF4545); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # dark by default

# ---- DARK theme tokens ----
DARK_CSS = """
<style>
:root, .stApp {
  --yb-bg: #111111;
  --yb-text: #F5F5F5;
  --yb-muted: #A3A3A3;
  --yb-surface: #181818;
  --yb-surface-sidebar: #181818;
  --yb-surface-code: #181818;
  --yb-surface-tablist: #181818;
  --yb-input-bg: #222222;
  --yb-input-text: #F5F5F5;
  --yb-border: #333333;
  --yb-accent: #FF4545;
  --yb-accent-hover: #ff5a5a;
  --yb-success: #22c55e;
  --yb-primary-btn-bg: #FF4545;
  --yb-primary-btn-text: #ffffff;
  --yb-selected-tab-bg: #222222;
  --yb-selected-tab-text: #F5F5F5;
  --yb-cite-bg: #2a1414;
  --yb-cite-border: #5a2020;
  --yb-cite-text: #ff8a8a;
  --yb-download-bg: #2a1414;
  --yb-download-border: #5a2020;
  --yb-download-text: #ffb3b3;
  --yb-focus: #FF4545;
}
body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, .main, .block-container { background: #111111 !important; }
.block-container, .main, p, li, span, label, h1, h2, h3, h4, .stMarkdown { color: #F5F5F5 !important; }
.stTextInput input, .stTextArea textarea { background: #222222 !important; color: #F5F5F5 !important; border: 1px solid #333333 !important; }
.stChatMessage { background: #181818 !important; border: 1px solid #333333 !important; }
[data-testid="stChatInput"] textarea { background: #222222 !important; color: #F5F5F5 !important; -webkit-text-fill-color: #F5F5F5 !important; }
.stTabs [data-baseweb="tab"] { color: #A3A3A3 !important; }
.stTabs [data-baseweb="tab-list"] { background: #181818 !important; }
.stExpander { background: #181818 !important; border-color: #333333 !important; }
.stExpander summary, .stExpander p, .stExpander span { color: #F5F5F5 !important; }
.stRadio label, .stCheckbox label, .stSelectbox label { color: #F5F5F5 !important; }
.stButton > button { background: #181818 !important; color: #F5F5F5 !important; border: 1px solid #333333 !important; }
.stButton > button:hover { border-color: #FF4545 !important; }
.stDownloadButton > button { background: #2a1414 !important; color: #ffb3b3 !important; border: 1px solid #5a2020 !important; }
.stMetric { background: #181818 !important; border: 1px solid #333333 !important; }
.stMetric label, .stMetric div { color: #F5F5F5 !important; }
code, pre, [data-testid="stCodeBlock"] { background: #181818 !important; }
[data-testid="stCodeBlock"] code { color: #F5F5F5 !important; }
.stAlert { filter: brightness(.95); }
hr { border-color: #333333 !important; }
[data-testid="stSidebar"] { background: #181818 !important; }
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stRadio label { color: #F5F5F5 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { background: #222222 !important; color: #F5F5F5 !important; -webkit-text-fill-color: #F5F5F5 !important; border: 1px solid #333333 !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div { background: #222222 !important; color: #F5F5F5 !important; border-color: #333333 !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] span, [data-testid="stSidebar"] div[data-baseweb="select"] svg { color: #F5F5F5 !important; fill: #F5F5F5 !important; }
</style>
"""
if st.session_state.dark_mode:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    # light mode: override surfaces to the light palette (shared exceptions retained)
    st.markdown("""
<style>
:root, .stApp {
  --yb-bg: #fafafa;
  --yb-text: #111;
  --yb-muted: #595959;
  --yb-surface: #ffffff;
  --yb-surface-sidebar: #f5f5f5;
  --yb-surface-code: #f5f5f5;
  --yb-surface-tablist: #f0f0f0;
  --yb-input-bg: #ffffff;
  --yb-input-text: #111;
  --yb-border: #ddd;
  --yb-accent: #FF4545;
  --yb-accent-hover: #ff5a5a;
  --yb-success: #22c55e;
  --yb-primary-btn-bg: #FF4545;
  --yb-primary-btn-text: #ffffff;
  --yb-selected-tab-bg: #222222;
  --yb-selected-tab-text: #F5F5F5;
  --yb-cite-bg: #2a1414;
  --yb-cite-border: #5a2020;
  --yb-cite-text: #ff8a8a;
  --yb-download-bg: #2a1414;
  --yb-download-border: #5a2020;
  --yb-download-text: #ffb3b3;
  --yb-focus: #FF4545;
}
body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, .main { background: #fafafa !important; }
.block-container { background: #fafafa !important; }
.block-container, .main, p, li, span, label, h1, h2, h3, h4, .stMarkdown { color: #111 !important; }
.stTextInput input, .stTextArea textarea { background: #fff !important; color: #111 !important; border: 1px solid #ddd !important; }
.stChatMessage { background: #fff !important; border: 1px solid #ddd !important; }
[data-testid="stChatInput"] textarea { background: #fff !important; color: #111 !important; }
.stTabs [data-baseweb="tab-list"] { background: #f0f0f0 !important; }
.stExpander { background: #fff !important; border-color: #ddd !important; }
.stButton > button { background: #fff !important; color: #111 !important; border: 1px solid #ddd !important; }
.stMetric { background: #fff !important; border: 1px solid #ddd !important; }
code, pre, [data-testid="stCodeBlock"] { background: #f5f5f5 !important; }
[data-testid="stSidebar"] { background: #f5f5f5 !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { background: #fff !important; color: #111 !important; -webkit-text-fill-color: #111 !important; }
.yb-card, .yb-processing { background: #fff !important; border-color: #ddd !important; }
/* shared exceptions retained in light mode */
.stButton > button[kind="primary"] { background: #FF4545 !important; color: #fff !important; border: none !important; }
.stButton > button[kind="primary"]:hover { background: #ff5a5a !important; }
.stTabs [aria-selected="true"] { color: #F5F5F5 !important; background: #222222 !important; }
.stDownloadButton > button { background: #2a1414 !important; color: #ffb3b3 !important; border: 1px solid #5a2020 !important; }
.yb-cite { background: #2a1414 !important; border-color: #5a2020 !important; color: #ff8a8a !important; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
def state_owner() -> str | None:
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

def new_chat(name: str | None = None) -> str:
    chat_id = uuid.uuid4().hex[:8]
    st.session_state.store[chat_id] = ChatMessageHistory()
    n = len(st.session_state.chat_names) + 1
    st.session_state.chat_names[chat_id] = name or f"Chat {n}"
    st.session_state.active_chat = chat_id
    return chat_id


# ---------- Header: logo (center), controls (right) ----------
_hc_left, _hc_mid, _hc_right = st.columns([3, 4, 3], vertical_alignment='top')
with _hc_mid:
    st.markdown(
        '<div class="yb-header" style="border-bottom:none;margin-bottom:0;padding-bottom:.6rem;"></div>',
        unsafe_allow_html=True,
    )
with _hc_right:
    _u_col = st.columns([1])[0]
    with _u_col:
        on_landing = (st.session_state.get('auth_user') is None and not st.session_state.get('guest') and not st.session_state.get('show_auth'))
        if st.session_state.get('auth_user'):
            st.caption(f"\U0001F464 {st.session_state.auth_user}")
            if st.button('Logout', key='top_logout'):
                persist_user_state()
                st.session_state.auth_user = None
                st.session_state.guest = False
                st.session_state.guest_sid = None
                for k in ('store','chat_names','active_chat','video_docs','video_ids','video_meta','video_lang','notes','quiz_data','raw_urls_box','video_snippets','translated_blocks','dubbed_mp4'):
                    st.session_state.pop(k, None)
                st.rerun()
        elif True:
            pass

if st.session_state.get("guest") or st.session_state.get("auth_user"):
    _home_left = st.columns([3, 7])[0]
    with _home_left:
        if st.button("Go to home page", key="home_btn", help="Go to landing page"):
            st.session_state.auth_user = None
            st.session_state.guest = False
            st.session_state.guest_sid = None
            st.session_state.video_docs = []
            st.session_state.video_ids = []
            st.session_state.video_snippets = []
            st.session_state.video_meta = {}
            st.session_state.show_auth = False
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.rerun()

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
    if not st.session_state.get("show_auth"):
        st.markdown(
            '<div style="text-align:center;margin:2rem 0 1rem;">'
            '<div class="yb-logo" style="justify-content:center;font-size:3rem;color:#FF4545;">'
            '<span style="color:#FF4545;">&#9654;</span> YOUBO</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="yb-hero"><h1>Understand any video.</h1>'
                    '<p>Ask questions, explore ideas, and learn from YouTube videos. '
                    'Paste a link below to begin.</p></div>', unsafe_allow_html=True)
        _c1, _c2, _c3 = st.columns([1, 3, 1])
        with _c2:
            url = st.text_input("YouTube URL", key="landing_url", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
            if st.button("Analyze Video", type="primary", use_container_width=True):
                if url.strip():
                    st.session_state.raw_urls_box = url.strip()
                    st.session_state.show_auth = True
                    st.rerun()
                else:
                    st.warning("Paste a YouTube link first.")
            if st.button("Sign in / Create account", use_container_width=True):
                st.session_state.show_auth = True
                st.rerun()
            _chips = "".join(
                f'<span class="yb-chip">{c}</span>'
                for c in ["Summarize this video", "Explain the key concepts",
                          "Extract important timestamps", "Generate quiz questions"]
            )
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;justify-content:center;'
                f'gap:.5rem;margin-top:.75rem;">{_chips}</div>',
                unsafe_allow_html=True,
            )
        st.stop()
    if st.button("Back to home", key="gate_back"):
        st.session_state.show_auth = False
        st.rerun()
    st.markdown('<div class="yb-hero" style="padding:3rem 1rem 1.5rem;"><h1 style="font-size:2rem;">Sign in to continue</h1></div>', unsafe_allow_html=True)
    _a1, _a2, _a3 = st.columns([1, 2.2, 1])
    with _a2:
        with st.container(border=True):
            t_login, t_signup, t_guest = st.tabs(["Sign in", "Create account", "Guest"])
            with t_login:
                u = st.text_input("Username", key="li_user")
                p = st.text_input("Password", type="password", key="li_pass")
                if st.button("Sign in", type="primary", use_container_width=True):
                    if auth.verify(u, p):
                        st.session_state.auth_user = u.strip().lower()
                        st.session_state.restore_pending = st.session_state.auth_user
                        st.session_state.show_auth = False
                        st.rerun()
                    else:
                        st.error("Wrong username or password.")
            with t_signup:
                u = st.text_input("Username", key="su_user")
                p = st.text_input("Password", type="password", key="su_pass")
                if st.button("Create account", type="primary", use_container_width=True):
                    err = auth.signup(u, p)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.auth_user = u.strip().lower()
                        st.session_state.restore_pending = st.session_state.auth_user
                        st.session_state.show_auth = False
                        st.success("Account created")
                        st.rerun()
            with t_guest:
                st.markdown('<p class="yb-muted">Guest mode: everything works and your work survives refresh. Create an account to keep it across devices.</p>', unsafe_allow_html=True)
                last = st.session_state.get("last_guest_sid")
                import secrets as _secrets
                can_restore = bool(last and auth.guest_state_exists(last))
                if st.button("Restore my guest chats" if can_restore else "Continue as guest", type="primary", use_container_width=True):
                    sid = last if can_restore else _secrets.token_hex(8)
                    st.session_state.guest = True
                    st.session_state.guest_sid = sid
                    st.session_state.last_guest_sid = sid
                    try:
                        st.query_params["s"] = sid
                    except Exception:
                        pass
                    if can_restore:
                        st.session_state.restore_pending = "_guest_" + sid
                    st.session_state.show_auth = False
                    st.rerun()
    st.stop()

if st.session_state.get("restore_pending"):
    owner = st.session_state.restore_pending
    st.session_state.restore_pending = False
    saved = auth.load_state(owner) if isinstance(owner, str) and owner else {}
    if saved:
        apply_saved_state(saved)
    st.rerun()

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
                     ("video_lang", ""), ("video_snippets", []), ("translated_blocks", None), ("translated_lang", None), ("dubbed_mp4", None), ("dubbed_lang", None), ("notes", None), ("quiz", None),
                     ("quiz_data", None), ("quiz_done", False)]:
    if key not in st.session_state:
        st.session_state[key] = default



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
            st.markdown(f"**{meta.get('title', vid)}**")
            duration = meta.get("duration") or 0
            st.caption(f"{meta.get('uploader', '')} · {duration // 60}m {duration % 60}s")


def load_video_ids(video_ids: list, lang_code: str, label: str = "Loading videos...") -> bool:
    """Fetch transcripts + build index. Returns True on success."""
    all_docs, meta, langs = [], {}, []
    st.session_state.video_snippets = []
    with st.status(label, expanded=True) as status:
        st.write("Fetching transcripts...")
        for vid in video_ids:
            try:
                with utils.Timer(f"transcript {vid}"):
                    snippets, lang_name = utils.fetch_transcript(vid, lang_code)
                    for sn in snippets:
                        sn.setdefault("video", vid)
                    st.session_state.video_snippets.extend(snippets)
                    all_docs.extend(utils.transcript_to_docs(vid, snippets))
                langs.append(lang_name)
                meta[vid] = utils.get_video_metadata(vid)
            except Exception as e:
                utils.log_error(f"transcript {vid}", e)
                st.write(f"No captions for {vid} — transcribing audio locally (slower, one-time)...")
                try:
                    snippets, lang_name = utils.transcribe_audio(vid)
                    for sn in snippets:
                        sn.setdefault("video", vid)
                    st.session_state.video_snippets.extend(snippets)
                    all_docs.extend(utils.transcript_to_docs(vid, snippets))
                    langs.append(lang_name)
                    meta[vid] = utils.get_video_metadata(vid)
                except Exception as e2:
                    utils.log_error(f"audio fallback {vid}", e2)
                    st.warning(
                        f"Skipping {vid}: this video has no captions and could not be "
                        "transcribed. Free workaround: use a video that already has "
                        "captions (most public uploads include auto-captions). "
                        f"Details: {e2}")
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
        st.error(
            "No transcripts available. None of the pasted videos have usable "
            "captions, and free transcription is unavailable on this PC "
            "(local Whisper is blocked by WDAC; cloud ASR needs HF credits). "
            "Try videos that already have captions.")
        return False
    st.session_state.video_docs = all_docs
    st.session_state.video_ids = video_ids
    st.session_state.video_meta = meta
    st.session_state.video_lang = ", ".join(sorted(set(langs))) or "unknown"
    st.success(f"Loaded {len(video_ids)} video(s), {len(all_docs)} chunks (captions: {st.session_state.video_lang}).")
    persist_user_state()
    return True


if load_btn:
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
        "'This video' / 'the video' / 'it' always refers to one of the loaded "
        "video(s) above. Answer from the retrieved context and name the video. "
        "Only if the retrieved context is empty may you say the information is "
        "not available. Never invent video counts, links, or playlist contents."
    )
    cite_rules = (
        "Cite every factual claim with the exact [Source](url) markdown link found "
        "in the retrieved context. The links include a timestamp (t=seconds); use "
        "them to answer 'what happens at X seconds' questions. Never invent "
        "timestamps, never use any other citation format (no brackets like ??, no footnotes)."
    )
    if st.session_state.get("compare_mode"):
        system_prompt = (
            "You compare what different videos say. "
            f"{scope} Compare and contrast perspectives across the loaded videos. "
            f"{cite_rules} Answer concretely from the retrieved context.\n\n{context}"
        )
    else:
        system_prompt = (
            "You answer questions directly about YouTube video content. "
            f"{scope} Answer helpfully and concretely from the retrieved context; "
            "no hedging, no 'as an AI', no refusing when context is present. "
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
    chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # ??? Two-column workspace ???????????????????????????????????????????????
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        # Video player + meta
        with st.container(border=True):
            if st.session_state.video_ids:
                vid = st.session_state.video_ids[0]
                import streamlit.components.v1 as _components
                _player_html = (
                    "<div id='yb-player'></div>"
                    "<script>"
                    "var tag=document.createElement('script');tag.src='https://www.youtube.com/iframe_api';"
                    "var fs=document.getElementsByTagName('script')[0];fs.parentNode.insertBefore(tag,fs);"
                    "var ybPlayer=null;"
                    "function onYouTubeIframeAPIReady(){"
                    "  ybPlayer=new YT.Player('yb-player',{height:'360',width:'100%',videoId:'" + vid + "',"
                    "    playerVars:{playsinline:1,rel:0}});"
                    "  window.parent.ybSeek=function(sec){if(ybPlayer&&ybPlayer.seekTo){ybPlayer.seekTo(sec,true);ybPlayer.playVideo();}};"
                    "  window.parent.ybPlayerReady=true;"
                    "}"
                    "</script>"
                )
                _components.html(_player_html, height=380)
                st.caption(f"[Open on YouTube](https://www.youtube.com/watch?v={vid})")
            meta0 = st.session_state.video_meta.get(st.session_state.video_ids[0], {}) if st.session_state.video_ids else {}
            if meta0:
                st.markdown(f"**{meta0.get('title', '')}**")
                st.caption(f"{meta0.get('uploader','')} ? {(meta0.get('duration') or 0)//60}m {(meta0.get('duration') or 0)%60}s ? {len(st.session_state.video_ids)} video(s) ? {st.session_state.video_lang}")
        # Research tabs
        t_summary, t_transcript, t_notes, t_dub = st.tabs(["Summary", "Transcript", "Notes", "Dub"])
        main_type = content_types[0] if content_types else "general"
        with t_summary:
            level = st.radio("Summary length", ["TL;DR", "Short", "Detailed"], horizontal=True, key="sum_level")
            if st.button("Generate summary", type="primary"):
                full_text = "\n".join(d.page_content for d in docs)
                with st.spinner("Summarizing..."):
                    try:
                        st.session_state.summary = utils.build_summary(llm, full_text, level, main_type)
                    except Exception as e:
                        utils.log_error("summary", e)
                        st.error("Summary failed.")
            if st.session_state.get("summary"):
                st.markdown(st.session_state.summary)
                st.download_button("Download summary (.md)", st.session_state.summary, file_name="summary.md", mime="text/markdown")
            st.markdown('<div class="yb-section-h">Key moments</div>', unsafe_allow_html=True)
            for m in utils.build_key_moments(docs):
                st.markdown(f"[{int(m['start'])}s]({m['link']}) ? {m['snippet']}")
        with t_transcript:
            q = st.text_input("Search transcript", key="tr_search", placeholder="keyword...")
            shown = 0
            snippets = st.session_state.get("video_snippets", [])
            segs_html = []
            for sn in snippets:
                txt = " ".join((sn.get("text") or "").split())
                if not txt:
                    continue
                if q and q.lower() not in txt.lower():
                    continue
                ts = sn.get("start")
                if ts is not None:
                    ts_i = int(ts)
                    ts_html = ("<a href='#' onclick='if(window.parent.ybSeek)window.parent.ybSeek(" + str(ts_i) + ");return false;' "
                               "style='color:#A3A3A3;text-decoration:none;margin-right:.5rem;'>[" + str(ts_i) + "s]</a>")
                else:
                    ts_html = ""
                import html as _html
                segs_html.append(f"<div style='margin:.35rem 0;line-height:1.5;'>{ts_html}{_html.escape(txt)}</div>")
                shown += 1
                if shown >= 120:
                    segs_html.append("<div style='color:#A3A3A3;font-size:.85rem;'>Showing first 120 matching segments.</div>")
                    break
            if segs_html:
                import streamlit.components.v1 as _components
                _components.html(
                    "<div style='font-family:Inter,system-ui,sans-serif;color:#F5F5F5;font-size:.92rem;max-height:460px;overflow-y:auto;padding:.25rem .5rem;'>"
                    + "".join(segs_html) +
                    "</div>",
                    height=480,
                )
            if shown == 0:
                st.info("No transcript segments match.")
            if shown == 0:
                st.info("No transcript segments match.")
        with t_notes:
            if st.button("Generate study notes", type="primary"):
                full_text = "\n".join(d.page_content for d in docs)
                with st.spinner("Summarizing..."):
                    try:
                        st.session_state.notes = utils.build_notes(llm, full_text, main_type)
                    except Exception as e:
                        utils.log_error("notes", e)
                        st.error("Notes generation failed.")
            if st.session_state.get("notes"):
                st.markdown(st.session_state.notes)
                c1, c2 = st.columns(2)
                c1.download_button("Download notes (.md)", st.session_state.notes, file_name="study_notes.md", mime="text/markdown")
                try:
                    c2.download_button("Download notes (.pdf)", utils.notes_to_pdf(st.session_state.notes), file_name="study_notes.pdf", mime="application/pdf")
                except Exception:
                    c2.info("Install fpdf2 for PDF export.")
            # Translate
            st.markdown('<div class="yb-section-h">Translate</div>', unsafe_allow_html=True)
            target = st.selectbox("Target language", utils.TARGET_LANGS, index=0, key="tr_lang")
            c1, c2 = st.columns(2)
            if c1.button("Translate summary", disabled=not st.session_state.get("summary")):
                with st.spinner(f"Translating summary to {target}..."):
                    try:
                        st.session_state.summary_tr = utils.translate_text(llm, st.session_state.summary, target)
                    except Exception as e:
                        utils.log_error("translate summary", e)
                        st.error("Translation failed.")
            if c2.button("Translate notes", disabled=not st.session_state.get("notes")):
                with st.spinner(f"Translating notes to {target}..."):
                    try:
                        st.session_state.notes_tr = utils.translate_text(llm, st.session_state.notes, target)
                    except Exception as e:
                        utils.log_error("translate notes", e)
                        st.error("Translation failed.")
            if st.session_state.get("summary_tr"):
                st.markdown(f"**Summary ({target})**")
                st.markdown(st.session_state.summary_tr)
            if st.session_state.get("notes_tr"):
                st.markdown(f"**Notes ({target})**")
                st.markdown(st.session_state.notes_tr)
            # Quiz
            st.markdown('<div class="yb-section-h">Quiz</div>', unsafe_allow_html=True)
            n_q = st.slider("Number of questions", 3, 10, 5, key="nq")
            if st.button("Generate quiz", type="primary"):
                full_text = "\n".join(d.page_content for d in docs)
                with st.spinner("Creating quiz..."):
                    try:
                        st.session_state.quiz_data = utils.build_quiz_data(llm, full_text, main_type, n=n_q)
                        st.session_state.quiz_done = False
                    except Exception as e:
                        utils.log_error("quiz", e)
                        st.error("Quiz generation failed.")
            if st.session_state.get("quiz_data"):
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    st.radio(f"q{i}", q["options"], index=None, key=f"quiz_a_{i}", label_visibility="collapsed")
                    if st.session_state.get("quiz_done"):
                        picked = st.session_state.get(f"quiz_a_{i}")
                        correct = picked == q["options"][q["answer"]]
                        st.markdown("Correct!" if correct else f"Wrong ? answer: **{q['options'][q['answer']]}**")
                        st.caption(q["explanation"])
                c1, c2 = st.columns(2)
                if c1.button("Check answers"):
                    st.session_state.quiz_done = True
                    st.rerun()
                if st.session_state.get("quiz_done"):
                    score = sum(1 for i, q in enumerate(st.session_state.quiz_data)
                                if st.session_state.get(f"quiz_a_{i}") == q["options"][q["answer"]])
                    c2.metric("Score", f"{score}/{len(st.session_state.quiz_data)}")

    with t_dub:
        st.markdown('<div class="yb-section-h">Translate / Dub video</div>', unsafe_allow_html=True)
        dub_lang = st.selectbox("Target language", utils.TARGET_LANGS, index=0, key="dub_lang")
        c1, c2 = st.columns(2)
        if c1.button("Translate transcript", type="primary"):
            with st.spinner(f"Translating transcript to {dub_lang}..."):
                try:
                    blocks = utils.translate_transcript(llm, docs, dub_lang, progress_cb=None)
                    st.session_state.translated_blocks = blocks
                    st.session_state.translated_lang = dub_lang
                except Exception as e:
                    utils.log_error("translate transcript", e)
                    st.error(f"Translation failed: {e}")
        if c2.button("Dub video (download MP4)", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            def _cb(f, msg=""):
                progress_bar.progress(min(1.0, float(f)))
                if msg:
                    status_text.caption(msg)
            try:
                mp4 = utils.dub_video(st.session_state.video_ids[0], llm, dub_lang, progress_cb=_cb)
                st.session_state.dubbed_mp4 = mp4
                st.session_state.dubbed_lang = dub_lang
                st.success("Dubbing complete.")
            except Exception as e:
                utils.log_error("dub video", e)
                st.error(f"Dubbing failed: {e}")
        if st.session_state.get("translated_blocks"):
            st.markdown(f"**Translated transcript ({st.session_state.get('translated_lang','')})**")
            for b in st.session_state.translated_blocks:
                ts = b.get("start", 0)
                link = b.get("link", "")
                label = f"[{int(ts)}s]({link})" if link else f"[{int(ts)}s]"
                st.markdown(f"{label} {b.get('text','')}")
        if st.session_state.get("dubbed_mp4"):
            st.markdown('<div class="yb-section-h">Dubbed preview</div>', unsafe_allow_html=True)
            st.video(st.session_state.dubbed_mp4)
            st.download_button(
                f"Download dubbed video ({st.session_state.get('dubbed_lang','')})",
                st.session_state.dubbed_mp4,
                file_name=f"dubbed_{st.session_state.get('dubbed_lang','video')}.mp4",
                mime="video/mp4",
            )

    with right:
        st.markdown('<div class="yb-section-h">Ask about this video</div>', unsafe_allow_html=True)
        history = get_session_history(st.session_state.active_chat)
        chat_box = st.container(height=420, border=True)
        with chat_box:
            if not history.messages:
                st.markdown('<div class="yb-empty"><h3>Start a conversation</h3>'
                            '<p>Ask anything about the loaded video.</p></div>', unsafe_allow_html=True)
                for sq in ["What is the main argument of this video?",
                           "Explain the key concepts in simple terms.",
                           "What examples does the speaker give?",
                           "Summarize the video in five bullet points."]:
                    if st.button(sq, key=f"sq_{sq[:12]}"):
                        st.session_state._pending_q = sq
                        st.rerun()
            for i, msg in enumerate(history.messages):
                with st.chat_message("user" if i % 2 == 0 else "assistant"):
                    st.markdown(msg.content)
        pending = st.session_state.pop("_pending_q", None)
        user_input = st.chat_input("Ask about the video(s):")
        if pending:
            user_input = pending
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
                            config={"configurable": {"session_id": st.session_state.active_chat}},
                        )
                        with st.chat_message("assistant"):
                            st.markdown(response["answer"])
                    except Exception as e:
                        utils.log_error("chat answer", e)
                        st.error("Sorry, that answer failed. See Diagnostics in the sidebar.")
                st.rerun()

    persist_user_state()

else:
    # Empty state for guests/accounts with no video
    st.markdown('<div class="yb-empty"><h3>Your workspace is ready</h3>'
                '<p>Load a video from the sidebar to unlock chat, summary, notes, and quiz.</p></div>',
                unsafe_allow_html=True)

