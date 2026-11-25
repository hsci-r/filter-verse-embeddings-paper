"""Text normalization helpers for lexical baselines."""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
from itertools import chain
import pandas as pd
BASE_NORMALIZE_RE = re.compile('[^0-9a-z\\s]+')
WHITESPACE_RE = re.compile('\\s+')

def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join((ch for ch in decomposed if not unicodedata.combining(ch)))

def basic_normalize(text: str) -> str:
    text = strip_diacritics(text).lower()
    text = BASE_NORMALIZE_RE.sub(' ', text)
    return WHITESPACE_RE.sub(' ', text).strip()

class AggressiveNormalizer:
    """Second normalization pass that removes train-rare characters."""

    def __init__(self, allowed_chars: set[str]):
        self.allowed_chars = allowed_chars | {' '}

    def __call__(self, text: str) -> str:
        normalized = basic_normalize(text)
        cleaned = ''.join((ch if ch in self.allowed_chars else ' ' for ch in normalized))
        return WHITESPACE_RE.sub(' ', cleaned).strip()

def iter_texts(series: pd.Series):
    for value in series.fillna(''):
        yield value

def build_allowed_chars(train_df: pd.DataFrame, min_char_count: int) -> tuple[set[str], dict[str, int]]:
    counter: Counter[str] = Counter()
    for text in chain(iter_texts(train_df['filter_text']), iter_texts(train_df['english_verse'])):
        counter.update(basic_normalize(text))
    allowed_chars = {ch for (ch, count) in counter.items() if ch != ' ' and count >= min_char_count}
    return (allowed_chars, dict(counter))
