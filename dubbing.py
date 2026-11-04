"""Dubbing: captions -> translation -> voice-over -> muxed MP4."""
import os
import subprocess
import tempfile

import prompts
from youtube import fetch_transcript

TTS_VOICES = {
    "Hindi": "hi-IN-SwaraNeural", "Spanish": "es-ES-ElviraNeural",
    "French": "fr-FR-DeniseNeural", "German": "de-DE-KatjaNeural",
    "Portuguese": "pt-BR-FranciscaNeural", "Arabic": "ar-SA-ZariyahNeural",
    "Tamil": "ta-IN-PallaviNeural", "Telugu": "te-IN-ShrutiNeural",
    "Bengali": "bn-IN-TanishaaNeural", "Marathi": "mr-IN-AarohiNeural",
    "Japanese": "ja-JP-NanamiNeural", "Korean": "ko-KR-SunHiNeural",
    "English": "en-US-AriaNeural",
}

MAX_DUB_SECONDS = 20 * 60


def dub_video(video_id: str, llm, target_lang: str, source_lang: str = "auto",
              progress_cb=None) -> bytes:
    """Captions -> per-window translation -> AI voice-over -> muxed with the
    original picture. Returns MP4 bytes. Voice-over style, not lip-synced."""
    import yt_dlp
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    voice = TTS_VOICES.get(target_lang)
    if not voice:
        raise ValueError(f"No voice for {target_lang}.")

    def progress(f, msg=""):
        if progress_cb:
            progress_cb(f, msg)

    progress(0.02, "Getting captions...")
    snippets, _ = fetch_transcript(video_id, source_lang)

    total_len = (snippets[-1]["start"] + snippets[-1].get("duration", 0)) if snippets else 0
    if total_len > MAX_DUB_SECONDS:
        raise RuntimeError(
            f"This video is {int(total_len // 60)} min — dubbing is capped at "
            f"{MAX_DUB_SECONDS // 60} min. Translate the transcript text instead.")

    # Group captions into roughly 12-second voice-over windows.
    windows, cur = [], []
    for s in snippets:
        cur.append(s)
        span = (s["start"] + s.get("duration", 0)) - cur[0]["start"]
        chars = sum(len(x["text"]) for x in cur)
        if span >= 12 or chars >= 400:
            windows.append(cur)
            cur = []
    if cur:
        windows.append(cur)

    tmpdir = tempfile.mkdtemp(prefix="youbo_dub_")
    try:
        import asyncio
        import edge_tts

        async def _tts_all(items):
            async def one(text, path):
                await edge_tts.Communicate(text, voice).save(path)
            await asyncio.gather(*[one(t, p) for t, p in items])

        progress(0.08, "Translating...")
        jobs = []
        for n, w in enumerate(windows, 1):
            src = " ".join(s["text"] for s in w)
            tr = llm.invoke(prompts.translate_prompt(src, target_lang, max_chars=2000)).content.strip()
            jobs.append((tr, os.path.join(tmpdir, f"w{n:03d}.mp3")))
            progress(0.08 + 0.30 * n / len(windows), f"Translating... {n}/{len(windows)}")

        progress(0.40, "Recording voiceover...")
        asyncio.run(_tts_all(jobs))
        progress(0.65, "Downloading original video...")
        with yt_dlp.YoutubeDL({
            "format": "bv*[height<=720]+ba/b[height<=720]/b",
            "quiet": True, "merge_output_format": "mp4",
            "ffmpeg_location": ffmpeg,
            "outtmpl": os.path.join(tmpdir, "src.%(ext)s"),
        }) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            src_path = ydl.prepare_filename(info)
            if not os.path.exists(src_path):
                base, _ = os.path.splitext(src_path)
                src_path = base + ".mp4"

        progress(0.75, "Mixing voiceover with picture...")
        inputs, filters, mix = [], [], []
        for n, (tr, path) in enumerate(jobs):
            inputs += ["-i", path]
            delay = int(windows[n][0]["start"] * 1000)
            filters.append(f"[{n}:a]adelay={delay}|{delay}[a{n}]")
            mix.append(f"[a{n}]")
        filters.append(f"{''.join(mix)}amix=inputs={len(jobs)}:normalize=0[aout]")
        dubbed = os.path.join(tmpdir, "dubbed.m4a")
        subprocess.run([ffmpeg, "-y", *inputs, "-filter_complex", ";".join(filters),
                        "-map", "[aout]", "-c:a", "aac", dubbed],
                       check=True, capture_output=True)
        progress(0.88, "Final video...")
        out_path = os.path.join(tmpdir, "dubbed.mp4")
        subprocess.run([ffmpeg, "-y", "-i", src_path, "-i", dubbed,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-shortest", out_path],
                       check=True, capture_output=True)
        progress(1.0, "Done.")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
