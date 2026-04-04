"""
text_utils.py
-------------
Utility helpers for:
  • Extracting text from PDF files (pure Python fallback chain)
  • Cleaning / normalising raw text
  • Chunking long documents
"""

import re
import unicodedata
from pathlib import Path
from typing import Union


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extract all text from a PDF file.

    Tries PyMuPDF (fitz) first for best quality, then falls back to
    pdfminer.six, and finally to pypdf.  Raises ImportError if none
    are installed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = ""

    # ── Strategy 1: PyMuPDF (fastest, best layout) ────────────────────────
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        text = "\n".join(pages)
        doc.close()
        if text.strip():
            return clean_text(text)
    except ImportError:
        pass

    # ── Strategy 2: pdfminer.six ──────────────────────────────────────────
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(pdf_path))
        if text.strip():
            return clean_text(text)
    except ImportError:
        pass

    # ── Strategy 3: pypdf ─────────────────────────────────────────────────
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        if text.strip():
            return clean_text(text)
    except ImportError:
        pass

    raise ImportError(
        "No PDF library found. Install one of: PyMuPDF, pdfminer.six, pypdf\n"
        "  pip install PyMuPDF"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize unicode, remove control characters, collapse whitespace.
    Preserves alphanumeric content and common punctuation.
    """
    # Normalize unicode (e.g. ligatures, accented chars)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove control / non-printable characters (except newline + tab)
    text = re.sub(r"[^\x09\x0a\x20-\x7e]", " ", text)

    # Collapse multiple blank lines into a single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def normalize_for_model(text: str) -> str:
    """
    Aggressive normalization for feeding into the tokenizer.
    Converts to lowercase and removes non-essential punctuation.
    """
    text = clean_text(text).lower()
    # Keep only alphanumeric, +, #, ., / (for skill names like C++, .NET)
    text = re.sub(r"[^a-z0-9#+./\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_words: int = 256, stride: int = 128) -> list[str]:
    """
    Split long text into overlapping word-level chunks.

    Parameters
    ----------
    text      : input string
    max_words : maximum number of words per chunk
    stride    : step size (overlap = max_words - stride)

    Returns a list of chunk strings.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride
    return chunks


def truncate_text(text: str, max_words: int = 256) -> str:
    """Return the first `max_words` words of text."""
    return " ".join(text.split()[:max_words])


# ─────────────────────────────────────────────────────────────────────────────
# 4. SECTION EXTRACTION (resume-specific)
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_HEADERS = {
    "skills":      r"\b(skills|technical skills|core competencies|technologies)\b",
    "experience":  r"\b(experience|work experience|employment|professional experience)\b",
    "education":   r"\b(education|academic|qualifications)\b",
    "projects":    r"\b(projects|personal projects|portfolio)\b",
    "summary":     r"\b(summary|profile|objective|about)\b",
}


def extract_section(text: str, section: str) -> str:
    """
    Attempt to extract a named section from a resume / document.

    Returns empty string if the section header is not found.
    """
    pattern = _SECTION_HEADERS.get(section)
    if not pattern:
        return ""

    lines = text.split("\n")
    in_section = False
    collected = []

    for line in lines:
        if re.search(pattern, line, re.IGNORECASE):
            in_section = True
            continue
        if in_section:
            # Stop at the next section header
            is_new_header = any(
                re.search(p, line, re.IGNORECASE)
                for p in _SECTION_HEADERS.values()
            )
            if is_new_header and collected:
                break
            collected.append(line)

    return "\n".join(collected).strip()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    SUMMARY
    Experienced Data Scientist with 5 years in ML.

    SKILLS
    Python, PyTorch, SQL, Docker, AWS, Machine Learning

    EXPERIENCE
    Senior Data Scientist at XYZ Corp (2020-2024)
    - Built recommendation systems using collaborative filtering
    - Deployed models on AWS SageMaker
    """
    print("Cleaned:\n", clean_text(sample)[:200])
    print("\nSkills section:\n", extract_section(sample, "skills"))
    print("\nChunks:", len(chunk_text(sample, max_words=20, stride=10)))
