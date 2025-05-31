import re
import snowballstemmer

def normalize(text: str) -> str:
    # lower-case, keep letters, digits, and spaces; remove everything else
    s = text.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_similarity(a: str, b: str) -> float:
    """
    a, b: already normalized, e.g. "reduced fat milk"
    returns percentage of query tokens found in b.
    """
    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens:
        return 0
    a_stems = {stemmer.stemWord(tok) for tok in a_tokens}
    b_stems = {stemmer.stemWord(tok) for tok in b_tokens}

    return len(a_stems & b_stems)
