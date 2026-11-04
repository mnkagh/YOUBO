import logging
import time
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


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
