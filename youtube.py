"""YouTube side: video IDs, playlists, metadata, captions, transcription."""
import os
import re

from logutil import log, Timer

MAX_VIDEOS = 20


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
        with yt_dlp.YoutubeDL({
            "format": "bestaudio/best", "quiet": True,
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
        }) as ydl:
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
    """Returns (snippets, language_name). Prefers the requested language,
    else manual captions, else auto-generated."""
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
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
