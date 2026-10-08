"""Local accounts + per-user saved state. Host-only file storage, no cloud."""
import hashlib
import json
import secrets
from pathlib import Path

DATA = Path(__file__).parent / "data"
USERS_FILE = DATA / "users.json"


def _load_users() -> dict:
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_users(users: dict) -> None:
    DATA.mkdir(exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def signup(username: str, password: str) -> str | None:
    """Returns error message, or None on success."""
    username = username.strip().lower()
    if not (3 <= len(username) <= 20 and username.replace("_", "").replace("-", "").isalnum()):
        return "Username: 3-20 chars, letters/numbers/_/- only."
    if len(password) < 4:
        return "Password: at least 4 characters."
    users = _load_users()
    if username in users:
        return "That username is taken."
    salt = secrets.token_hex(16)
    users[username] = {"salt": salt, "hash": _hash(password, salt)}
    _save_users(users)
    return None


def verify(username: str, password: str) -> bool:
    users = _load_users()
    u = users.get(username.strip().lower())
    return bool(u) and _hash(password, u["salt"]) == u["hash"]


def user_dir(username: str) -> Path:
    d = DATA / "users" / username.strip().lower()
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def save_state(username: str, state: dict) -> None:
    (user_dir(username) / "state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8")


def load_state(username: str) -> dict:
    try:
        return json.loads((user_dir(username) / "state.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def guest_state_exists(token: str) -> bool:
    return (DATA / "users" / f"_guest_{token}" / "state.json").exists()


def prune_old_guests(max_age_days: int = 14) -> int:
    """Delete anonymous guest states older than N days. Returns count removed."""
    import time
    removed = 0
    users_dir = DATA / "users"
    if not users_dir.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    for d in users_dir.iterdir():
        if d.is_dir() and d.name.startswith("_guest_"):
            try:
                if d.stat().st_mtime < cutoff:
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except OSError:
                pass
    return removed
