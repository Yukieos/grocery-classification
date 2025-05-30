import re
from difflib import SequenceMatcher

def normalize(text: str) -> str:
    return ''.join(
        c.lower() if (c.isalnum() or c.isspace()) else ''
        for c in text
    ).strip()


def compute_similarity(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)
