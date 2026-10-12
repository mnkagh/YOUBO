"""Accounts + per-user app state in SQLite (data/youbо.db). Old JSON files migrate on first use."""
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

DATA = Path(__file__).parent / "data"
DB_FILE = DATA / "youbо.db"
_init_lock = threading.Lock()
_migrated = False


def _db() -> sqlite3.Connection:
    global _migrated
    DATA.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_FILE, timeout=20)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, salt TEXT NOT NULL, "
              "hash TEXT NOT NULL, created_at REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS state(owner TEXT PRIMARY KEY, data TEXT NOT NULL, "
              "updated_at REAL NOT NULL)")
    c.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, val TEXT)")
    if not _migrated:
        with _init_lock:
            if not _migrated:
                _migrate(c)
                _migrated = True
    return c


def _migrate(c: sqlite3.Connection) -> None:
    """One-time import of the legacy users.json + data/users/<owner>/state.json files."""
    if c.execute("SELECT 1 FROM meta WHERE key='migrated'").fetchone():
        return
    now = time.time()
    users_file = DATA / "users.json"
    if users_file.exists():
        try:
            users = json.loads(users_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            users = {}
        for name, u in users.items():
            c.execute("INSERT OR IGNORE INTO users(username, salt, hash, created_at) VALUES (?,?,?,?)",
                      (name.strip().lower(), u.get("salt", ""), u.get("hash", ""), now))
    users_dir = DATA / "users"
    if users_dir.exists():
        for d in users_dir.iterdir():
            f = d / "state.json"
            if not (d.is_dir() and f.exists()):
                continue
            try:
                payload = f.read_text(encoding="utf-8")
                json.loads(payload)
                mtime = f.stat().st_mtime
            except (OSError, json.JSONDecodeError):
                continue
            c.execute("INSERT OR REPLACE INTO state(owner, data, updated_at) VALUES (?,?,?)",
                      (d.name, payload, mtime))
    c.execute("INSERT OR REPLACE INTO meta(key, val) VALUES ('migrated', '1')")
    c.commit()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def signup(username: str, password: str) -> str | None:
    """Returns error message, or None on success."""
    username = username.strip().lower()
    if not (3 <= len(username) <= 20 and username.replace("_", "").replace("-", "").isalnum()):
        return "Username: 3-20 chars, letters/numbers/_/- only."
    if len(password) < 4:
        return "Password: at least 4 characters."
    salt = secrets.token_hex(16)
    with closing(_db()) as c, c:
        if c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            return "That username is taken."
        c.execute("INSERT INTO users(username, salt, hash, created_at) VALUES (?,?,?,?)",
                  (username, salt, _hash(password, salt), time.time()))
    return None


def verify(username: str, password: str) -> bool:
    with closing(_db()) as c:
        row = c.execute("SELECT salt, hash FROM users WHERE username=?",
                        (username.strip().lower(),)).fetchone()
    return bool(row) and _hash(password, row[0]) == row[1]


def history_to_list(history) -> list:
    return [{"role": ("human" if m.type == "human" else "ai"), "content": m.content}
            for m in history.messages]


def list_to_history(items: list):
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    h = ChatMessageHistory()
    for item in items:
        h.add_message(HumanMessage(content=item["content"]) if item["role"] == "human"
                      else AIMessage(content=item["content"]))
    return h


def save_state(owner: str, state: dict) -> None:
    payload = json.dumps(state, ensure_ascii=False)
    with closing(_db()) as c, c:
        c.execute("INSERT INTO state(owner, data, updated_at) VALUES (?,?,?) "
                  "ON CONFLICT(owner) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                  (owner, payload, time.time()))


def load_state(owner: str) -> dict:
    with closing(_db()) as c:
        row = c.execute("SELECT data FROM state WHERE owner=?", (owner,)).fetchone()
    try:
        return json.loads(row[0]) if row else {}
    except json.JSONDecodeError:
        return {}


def guest_state_exists(token: str) -> bool:
    with closing(_db()) as c:
        return c.execute("SELECT 1 FROM state WHERE owner=?",
                         (f"_guest_{token}",)).fetchone() is not None


def prune_old_guests(max_age_days: int = 14) -> int:
    """Delete anonymous guest states older than N days. Returns count removed."""
    cutoff = time.time() - max_age_days * 86400
    with closing(_db()) as c, c:
        return c.execute("DELETE FROM state WHERE owner GLOB '_guest_*' AND updated_at < ?",
                         (cutoff,)).rowcount
