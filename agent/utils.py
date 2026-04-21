"""
Utility functions for text normalization and parsing.
"""
import json
import re
import string
from typing import Optional

# Regex patterns
_Q_RE = re.compile(
    r"\b(?:(?P<year>\d{4})\s*[/\-\s]?\s*(?:q|quarter)\s*(?P<q>\d{1})\b"
    r"|(?P<ord>first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+of\s+(?P<year2>\d{4})\b)\b",
    re.I,
)

_LAYER_RE = re.compile(
    r"\b(?:switch|change|toggle)\s+(?:to\s+)?(?P<layer>[a-z0-9][a-z0-9\s\-/]*?)\s*(?:layer|view)?\b",
    re.I,
)


def _split_camel_and_snake(s: str) -> str:
    """Split camelCase and snake_case into spaces."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ")
    return s


def _norm(text: str) -> str:
    """Normalize text for matching: lowercase, remove punctuation, collapse whitespace."""
    text = text or ""
    text = _split_camel_and_snake(text).lower()
    text = re.sub(r"[\(\[].*?[\)\]]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _stringify(v) -> str:
    """Convert value to string, handling None and floats."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def _normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    s = _stringify(s).casefold()
    s = re.sub(r"[\u2010-\u2015]", "-", s)  # normalize unicode hyphens
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canonical_quarter(s: str) -> Optional[str]:
    """
    Returns canonical 'YYYY Qn' if s looks like a quarter expression.
    Handles: '2024 Quarter 2', '2024/Q2', 'second quarter of 2024', etc.
    """
    s0 = _stringify(s)
    m = _Q_RE.search(s0)
    if not m:
        return None

    ordmap = {
        "first": "1", "1st": "1",
        "second": "2", "2nd": "2",
        "third": "3", "3rd": "3",
        "fourth": "4", "4th": "4",
    }

    if m.group("year"):
        return f"{m.group('year')} Q{m.group('q')}"
    if m.group("ord"):
        qn = ordmap.get(m.group("ord").lower(), m.group("ord"))
        return f"{m.group('year2')} Q{qn}"
    return None


def _extract_layer_name(user_query: str) -> Optional[str]:
    """Extract layer name from a switch/change request."""
    m = _LAYER_RE.search(user_query or "")
    if not m:
        return None
    layer = _normalize_text(m.group("layer"))
    if not layer:
        return None
    return layer.replace(" ", "_")


def _extract_single_x_value(user_query: str):
    """Extract a single x-axis value from the query (year, quarter, or number)."""
    if not user_query:
        return None

    # Prefer quarter canonicalisation
    qcanon = _canonical_quarter(user_query)
    if qcanon:
        return qcanon

    # Then year
    m = re.search(r"\b((?:19|20)\d{2})\b", user_query)
    if m:
        return m.group(1)

    # Then bare number
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", user_query)
    if m:
        return m.group(1)

    return None


def _extract_pan_numeric_factor(user_query: str) -> Optional[int]:
    """
    Extract pan factor from query.
    - "pan left 150%" -> 150
    - "pan left 200"  -> 200
    """
    q = user_query or ""
    m = re.search(r"\b(\d{2,3})\s*%\b", q)
    if m:
        return int(m.group(1))

    # only accept bare numbers if the utterance is clearly a pan instruction
    m = re.search(r"\bpan\b.*?\b(\d{2,3})\b", q, re.I)
    if m:
        return int(m.group(1))

    return None


def _extract_bulleted_items(text: str):
    """Extract bulleted list items from text."""
    lines = text.splitlines()
    bullet = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
    items, idxs = [], []
    for i, ln in enumerate(lines):
        m = bullet.match(ln)
        if m:
            items.append(m.group(1).strip())
            idxs.append(i)
    if not items:
        return text, [], -1, -1
    return "\n".join(lines[:idxs[0]]).rstrip(), items, idxs[0], idxs[-1]


def parse_llm_json(raw: str, fallback: dict) -> dict:
    """
    Parse a JSON response from an LLM, handling markdown code fences.
    Returns fallback dict if parsing fails.
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS reads clean text."""
    # Bold and italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Collapse all newlines to spaces for TTS
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def rewrite_long_lists_locally(text: str, max_per_sentence: int = 2, min_trigger: int = 4) -> str:
    """Rewrite long bulleted lists into prose for TTS."""
    prefix, items, s, e = _extract_bulleted_items(text)
    if len(items) < min_trigger:
        return text

    chunks = []
    for i in range(0, len(items), max_per_sentence):
        chunk = items[i:i + max_per_sentence]
        chunks.append(", ".join(chunk))

    prose = ". ".join(chunks) + "."
    return f"{prefix} {prose}" if prefix else prose
