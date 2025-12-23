"""Module for word list management in Spacewords."""

from collections import Counter, defaultdict
from pathlib import Path

from spacewords.solver.config import config as solver_config


def load_word_list(*, min_len: int = 2, max_len: int | None = None) -> set[str]:
    """Load the word list from the configured dictionary file.

    Args:
        min_len: Minimum word length to include (defaults to 2, since slots are length > 1).
        max_len: Optional maximum word length to include.

    Returns:
        A set of valid words.
    """
    word_list_path = Path(solver_config.word_list_path)
    if not word_list_path.is_file():
        raise FileNotFoundError(f"Word list file not found: {word_list_path}")

    with word_list_path.open("r", encoding="utf-8") as f:
        words: set[str] = set()
        for line in f:
            word = line.strip().upper()
            if not word:
                continue
            if len(word) < min_len:
                continue
            if max_len is not None and len(word) > max_len:
                continue
            words.add(word)
        return words


def create_word_map(words: set[str]) -> dict[tuple[int, int, str], set[str]]:
    """Create a mapping for a set of words.

    The index is `(word_length, position_in_word, character)`.  A word appears in the set
    for index `(L, p, c)` if it has length `L` and character `c` at position `p` (0-based).
    """
    word_map: defaultdict[tuple[int, int, str], set[str]] = defaultdict(set)

    for word in words:
        word_length = len(word)
        for position, char in enumerate(word):
            key = (word_length, position, char)
            word_map[key].add(word)

    return dict(word_map)  # Convert defaultdict to regular dict for return


def get_letter_frequency(words: set[str]) -> dict[str, float]:
    """Compute the frequency of each letter in the given set of words, as percentages.

    Args:
        words (set[str]): A set of words.
    """
    letter_freq = Counter()
    for word in words:
        letter_freq.update(word)
    total_letter_count = letter_freq.total()
    if total_letter_count == 0:
        return {}
    return {ch: freq / total_letter_count * 100 for ch, freq in letter_freq.items()}
