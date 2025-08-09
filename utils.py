from typing import List, Dict, Tuple
import hashlib
import os
import re

def trim_bytes(text: str, max_bytes: int = 3500) -> str:
    """Trim a string to fit under max_bytes in UTF-8."""
    if text is None:
        return ""
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(text[:mid].encode("utf-8")) <= max_bytes:
            lo = mid + 1
        else:
            hi = mid
    return text[:hi-1]

def flush_section(sections: List[Dict[str, str]], current_heading: str, buffer: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    """Append buffered text as a section and reset the buffer."""
    if current_heading is None:
        return sections, buffer
    text = "\n".join(buffer).strip()
    sections.append({"heading": (current_heading or "").strip(), "text": text})
    return sections, []

def is_table(block: str) -> bool:
    """Heuristic to detect Markdown-like tables."""
    rows = [r.strip() for r in block.strip().splitlines() if r.strip()]
    if len(rows) < 2:
        return False
    if "|" in rows[0] and "|" in rows[1] and re.search(r'\|?\s*:?-{3,}\s*\|', rows[1]):
        return True
    return sum(1 for r in rows[:3] if "|" in r) >= 2

def persist_chunk(text: str, store_dir: str) -> str:
    """Persist large chunk to disk in store_dir and return a stable key."""
    os.makedirs(store_dir, exist_ok=True)
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    path = os.path.join(store_dir, f"{key}.txt")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return key

def load_chunk(key: str, store_dir: str) -> str:
    """Load a persisted chunk by key from store_dir."""
    path = os.path.join(store_dir, f"{key}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""