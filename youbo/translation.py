"""Translation of transcripts and generated content."""
from . import prompts

TARGET_LANGS = ["Hindi", "Spanish", "French", "German", "Portuguese",
                "Arabic", "Tamil", "Telugu", "Bengali", "Marathi",
                "Japanese", "Korean", "English"]


def translate_text(llm, text: str, target_lang: str, max_chars: int = 12000) -> str:
    return llm.invoke(prompts.translate_prompt(text, target_lang, max_chars)).content


def translate_transcript(llm, docs: list, target_lang: str, batch_chunks: int = 4,
                         progress_cb=None) -> list:
    """Translate the whole video batch by batch. Each block keeps the
    timestamp link of where it starts, so the transcript reads in the target
    language with clickable timestamps."""
    blocks = []
    batches = [docs[i:i + batch_chunks] for i in range(0, len(docs), batch_chunks)]
    for n, batch in enumerate(batches, 1):
        text = "\n\n".join(d.page_content for d in batch)
        tr = llm.invoke(prompts.translate_prompt(text, target_lang, max_chars=15000)).content
        blocks.append({
            "start": batch[0].metadata.get("start", 0),
            "link": batch[0].metadata.get("link", ""),
            "video_id": batch[0].metadata.get("video_id", ""),
            "text": tr,
        })
        if progress_cb:
            progress_cb(n / len(batches))
    return blocks
