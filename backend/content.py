"""LLM-generated artefacts: summaries, notes, quizzes, PDF export."""
import json

from core import prompts


def build_summary(llm, transcript_text: str, level: str, content_type: str = "general", max_chars: int = 12000) -> str:
    prompt = prompts.summary_prompt(level, content_type).format(
        transcript=transcript_text[:max_chars])
    return llm.invoke(prompt).content


def build_notes(llm, transcript_text: str, content_type: str = "general", max_chars: int = 12000) -> str:
    prompt = prompts.notes_prompt(content_type).format(
        transcript=transcript_text[:max_chars])
    return llm.invoke(prompt).content


def build_quiz_data(llm, transcript_text: str, content_type: str = "general", n: int = 5, max_chars: int = 12000) -> list:
    """Structured quiz for the interactive UI: {question, options[4], answer, explanation}."""
    prompt = prompts.quiz_data_prompt(n, content_type).format(
        transcript=transcript_text[:max_chars])
    raw = llm.invoke(prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    clean = []
    for item in data[:n]:
        opts = [str(o) for o in item.get("options", [])][:4]
        ans = int(item.get("answer", 0))
        if len(opts) == 4 and 0 <= ans <= 3 and item.get("question"):
            clean.append({
                "question": str(item["question"]),
                "options": opts,
                "answer": ans,
                "explanation": str(item.get("explanation", "")),
            })
    if not clean:
        raise ValueError("Quiz came back empty — try again.")
    return clean


def notes_to_pdf(notes_md: str) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", size=11)
    for line in notes_md.splitlines():
        safe = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 7, safe)
    return bytes(pdf.output())
