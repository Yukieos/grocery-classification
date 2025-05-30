import re
from difflib import SequenceMatcher


def normalize(text: str) -> str:
    # lower-case, keep letters, digits, and spaces; remove everything else
    s = text.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    # collapse multiple spaces → one
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_similarity(a: str, b: str) -> float:
    """
    a, b: already normalized, e.g. "reduced fat milk"
    returns percentage of query tokens found in b.
    """
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)
