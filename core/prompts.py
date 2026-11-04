"""All LLM prompts in one place, as plain text builders."""


def _clip(model: dict, key: str) -> str:
    return model.get(key, next(iter(model.values())))


def summary_prompt(level: str, content_type: str) -> str:
    instructions = {
        "TL;DR": "Write a TL;DR summary in at most 3 sentences.",
        "Short": "Write a short summary in one paragraph (~5 sentences).",
        "Detailed": "Write a detailed summary with sections: Overview, Main Points, Conclusion.",
    }
    hint = {
        "music": "Focus on theme, mood, and message. ",
        "gaming": "Focus on what happens and key moments. ",
        "educational": "Focus on concepts taught. ",
        "general": "",
    }[content_type]
    text = (
        f"{hint}{_clip(instructions, level)}\n"
        "Respond in the same language as the transcript.\n\n{transcript}"
    )
    return text


def notes_prompt(content_type: str) -> str:
    style = {
        "music": "Turn this music video transcript/lyrics into fan-style notes: themes, standout lines, and vibe, in markdown.",
        "gaming": "Turn this gaming video transcript into notes: key plays, strategies, and highlights, in markdown.",
        "educational": "Turn this transcript into clean, structured study notes in markdown with headings, bullet points, and key takeaways.",
        "general": "Turn this video transcript into clean, structured notes in markdown with headings, bullet points, and key takeaways.",
    }[content_type]
    return f"{style}\nRespond in the same language as the transcript.\n\n{{transcript}}"


def quiz_data_prompt(n: int, content_type: str) -> str:
    flavor = {
        "music": "fun trivia about the song, artist, lyrics and theme",
        "gaming": "questions about plays, strategies and moments in the video",
        "educational": "questions testing understanding of the concepts taught",
        "general": "questions about the key points of the video",
    }[content_type]
    return (
        f"Based on the transcript below, write {n} multiple-choice questions ({flavor}). "
        "Reply with ONLY a valid JSON array, no markdown fences, no commentary. "
        'Each item: {"question": "...", "options": ["...", "...", "...", "..."], '
        '"answer": 0-3 index of the correct option, "explanation": "one line"}.\n'
        "Respond in the same language as the transcript.\n\n{transcript}"
    )


def translate_prompt(text: str, target_lang: str, max_chars: int = 12000) -> str:
    return (
        f"Translate the following markdown content into {target_lang}. "
        "Keep all markdown formatting, headings, bullet points and links intact. "
        "Reply with ONLY the translation.\n\n" + text[:max_chars]
    )


def rag_condense_system() -> str:
    return (
        "Given a chat history and the latest user question which might reference "
        "context in the chat history, formulate a standalone question. Do NOT answer it, "
        "just reformulate it if needed and otherwise return it as is."
    )


def rag_scope(video_titles: list, content_types: list) -> str:
    return (
        f"You know ONLY these loaded video(s): {'; '.join(video_titles)}. "
        f"Content type(s): {', '.join(content_types)}. "
        "'This video' / 'the video' / 'it' always refers to one of the loaded "
        "video(s) above. Answer from the retrieved context and name the video. "
        "Only if the retrieved context is empty may you say the information is "
        "not available. Never invent video counts, links, or playlist contents."
    )


RAQ_CITE_RULES = (
    "Cite every factual claim with the exact [Source](url) markdown link found "
    "in the retrieved context. The links include a timestamp (t=seconds); use "
    "them to answer 'what happens at X seconds' questions. Never invent "
    "timestamps, never use any other citation format (no brackets like ??, no footnotes)."
)


def rag_qa_system(compare_mode: bool, video_titles: list, content_types: list) -> str:
    scope = rag_scope(video_titles, content_types)
    cite = RAQ_CITE_RULES
    if compare_mode:
        return (
            "You compare what different videos say. "
            f"{scope} Compare and contrast perspectives across the loaded videos. "
            f"{cite} Answer concretely from the retrieved context.\n\n{{context}}"
        )
    return (
        "You answer questions directly about YouTube video content. "
        f"{scope} Answer helpfully and concretely from the retrieved context; "
        "no hedging, no 'as an AI', no refusing when context is present. "
        f"{cite} If the answer isn't in the context, say what you do know "
        "from the videos instead of refusing.\n\n{{context}}"
    )
