"""YouTube side: video IDs, playlists, metadata, captions, transcription."""
import os
import re
from pathlib import Path

from core.logutil import log, Timer

MAX_VIDEOS = 20


def cookie_file() -> str | None:
    """Netscape-format cookies.txt from the host, if present. Checked in
    order: YOUTUBE_COOKIES env, Render secret files, project root."""
    for p in (os.getenv("YOUTUBE_COOKIES", ""),
              "/etc/secrets/cookies.txt",
              str(Path(__file__).resolve().parent.parent / "cookies.txt")):
        if p and os.path.isfile(p):
            return p
    return None


def ydl_opts(extra: dict | None = None) -> dict:
    opts = {"quiet": True}
    cf = cookie_file()
    if cf:
        opts["cookiefile"] = cf
    if extra:
        opts.update(extra)
    return opts


def _http_session():
    import http.cookiejar
    import requests
    session = requests.Session()
    cf = cookie_file()
    if cf:
        jar = http.cookiejar.MozillaCookieJar(cf)
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(jar)
        except Exception:
            log.warning(f"could not load cookies from {cf}")
    return session


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
        with yt_dlp.YoutubeDL(ydl_opts({"extract_flat": True})) as ydl:
            info = ydl.extract_info(url, download=False)
        return [(e.get("url") or e.get("id")) for e in info.get("entries", [])[:limit] if e]
    except Exception:
        return []


def get_video_metadata(video_id: str) -> dict:
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts({"skip_download": True})) as ydl:
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


def transcribe_audio(video_id: str, model_name: str | None = None) -> tuple:
    """Fallback for videos without captions: download the audio and transcribe
    it (local Whisper, else free Hugging Face cloud ASR). Returns snippets in
    the same shape as captions."""
    import tempfile
    import yt_dlp
    tmpdir = tempfile.mkdtemp(prefix="youbo_")
    try:
        with yt_dlp.YoutubeDL(ydl_opts({
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
        })) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            audio_path = ydl.prepare_filename(info)
        try:
            return _transcribe_local(audio_path, model_name), "unknown (local speech-to-text)"
        except Exception as e:
            log.warning(f"local transcription blocked ({e}) — trying cloud ASR")
            try:
                return _transcribe_cloud(audio_path)
            except Exception as e2:
                raise RuntimeError(
                    "This video has no captions, and transcription is not available "
                    "right now. Local Whisper is blocked by Windows Defender "
                    "Application Control (WDAC) on this PC, and cloud ASR "
                    "needs Hugging Face credits. Free workaround: use a video "
                    "that already has captions (most public uploads have "
                    "auto-captions). "
                    f"(local: {type(e).__name__}; cloud: {e2})")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _transcribe_local(audio_path: str, model_name: str | None = None) -> list:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "Local transcription unavailable: faster-whisper could not load "
            f"({type(e).__name__}). On this PC Windows Defender Application "
            "Control (WDAC) blocks the native libraries Whisper needs.")
    model_name = model_name or os.getenv("WHISPER_MODEL", "tiny")
    with Timer(f"whisper local [{model_name}]"):
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=5)
        snippets = [{"text": s.text.strip(), "start": s.start, "duration": s.end - s.start}
                    for s in segments if s.text.strip()]
    if not snippets:
        raise RuntimeError("Audio transcription came back empty.")
    return snippets


def _transcribe_cloud(audio_path: str) -> tuple:
    from huggingface_hub import InferenceClient
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "This video has no captions, and on-device transcription is blocked on this PC. "
            "Add a free Hugging Face token as HF_TOKEN in .env to transcribe such videos "
            "(https://huggingface.co/settings/tokens).")
    size_mb = os.path.getsize(audio_path) / 1e6
    if size_mb > 25:
        raise RuntimeError(
            f"This video has no captions and its audio ({size_mb:.0f}MB) is too big for "
            "free cloud transcription. Try a shorter video.")
    with Timer("whisper cloud"):
        client = InferenceClient(token=token)
        with open(audio_path, "rb") as f:
            out = client.automatic_speech_recognition(
                f.read(), model="openai/whisper-large-v3-turbo",
                extra_body={"return_timestamps": True})
    chunks = getattr(out, "chunks", None) or (out.get("chunks") if isinstance(out, dict) else None) or []
    snippets = []
    for c in chunks:
        text = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
        ts = getattr(c, "timestamp", None) if not isinstance(c, dict) else c.get("timestamp")
        if not text or not text.strip():
            continue
        start = (ts[0] if ts else 0) or 0
        end = (ts[1] if ts and len(ts) > 1 else start) or start
        snippets.append({"text": " ".join(text.split()), "start": start,
                         "duration": max(0.0, end - start)})
    if not snippets:
        raise RuntimeError("Cloud transcription came back empty.")
    return snippets, "unknown (cloud speech-to-text)"


def fetch_transcript(video_id: str, language: str = "auto") -> tuple:
    """Returns (snippets, language_name). Tries the transcript API first,
    then yt-dlp caption download (helps when YouTube blocks the server IP)."""
    try:
        return _fetch_transcript_api(video_id, language)
    except Exception as e:
        log.warning(f"transcript api failed for {video_id}: {e}")
        return _fetch_transcript_ytdlp(video_id, language)


def _fetch_transcript_api(video_id: str, language: str = "auto") -> tuple:
    """Returns (snippets, language_name). Prefers the requested language,
    else manual captions, else auto-generated."""
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi(http_client=_http_session())
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
    snippets = [{"text": " ".join((s.text or "").split()), "start": s.start, "duration": s.duration} for s in data]
    return snippets, transcript.language


def _fetch_transcript_ytdlp(video_id: str, language: str = "auto") -> tuple:
    """Download the caption track with yt-dlp (cookies-aware) and parse it."""
    import glob
    import shutil
    import tempfile
    import yt_dlp
    langs = [language] if language not in ("auto", "") else []
    if "en" not in langs:
        langs.append("en")
    tmpdir = tempfile.mkdtemp(prefix="youbo_cap_")
    try:
        with yt_dlp.YoutubeDL(ydl_opts({
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": langs,
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(tmpdir, "cap.%(ext)s"),
        })) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        files = glob.glob(os.path.join(tmpdir, "cap*.vtt"))
        if not files:
            raise RuntimeError("yt-dlp wrote no caption track for this video.")
        snippets = _parse_vtt(files[0])
        if not snippets:
            raise RuntimeError("Caption file was empty.")
        tracks = info.get("subtitles") or info.get("automatic_captions") or {}
        lang_name = f"unknown ({next(iter(tracks), 'yt-dlp')} captions)" if tracks else "unknown (yt-dlp captions)"
        return snippets, lang_name
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_vtt(path: str) -> list:
    import io
    ts_re = re.compile(r"(\d+):(\d+):([\d.]+)\s*-->\s*(\d+):(\d+):([\d.]+)")
    tag_re = re.compile(r"<[^>]+>")
    snippets = []
    prev_lines = []
    start, dur, lines = 0.0, 0.0, []

    def flush():
        nonlocal lines, prev_lines
        if lines:
            if len(lines) > 1 and prev_lines and lines[0] == prev_lines[-1]:
                lines = lines[1:]
            text = " ".join(" ".join(lines).split())
            if text:
                snippets.append({"text": text, "start": start, "duration": dur})
            prev_lines = lines
        lines = []

    for line in io.open(path, encoding="utf-8", errors="replace").read().splitlines():
        m = ts_re.search(line)
        if m:
            flush()
            h1, m1, s1, h2, m2, s2 = m.groups()
            start = int(h1) * 3600 + int(m1) * 60 + float(s1)
            end = int(h2) * 3600 + int(m2) * 60 + float(s2)
            dur = max(0.0, end - start)
        elif "-->" not in line and not line.startswith(
                ("WEBVTT", "Kind:", "Language:", "Style:", "NOTE")) and line.strip():
            t = tag_re.sub("", line.strip())
            if t:
                lines.append(t)
    flush()
    return snippets
