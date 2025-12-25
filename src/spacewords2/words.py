"""Module for handling a word list."""

import pathlib

from bitarray import bitarray
from bitarray.util import ones, zeros
from sortedcontainers import SortedList

WORD_LIST_FILE = "spw-dict.txt"


def find_wordlist_file() -> str:
    """Iterate up the directory tree to find the word list file."""
    current_dir = pathlib.Path(__file__).parent
    while True:
        candidate = current_dir / WORD_LIST_FILE
        if candidate.is_file():
            return str(candidate)
        if current_dir.parent == current_dir:
            raise FileNotFoundError(f"Could not find {WORD_LIST_FILE} in any parent directory.")
        current_dir = current_dir.parent


WORDLIST_PATH = find_wordlist_file()


def load_wordlist() -> SortedList[str]:
    """Load the word list from the file."""
    with open(WORDLIST_PATH, "r") as f:
        words = [stripped.upper() for line in f if (stripped := line.strip())]
    print(f"Loaded {len(words)} words from {WORDLIST_PATH}")
    return SortedList(words)


WORD_LIST = load_wordlist()
MAX_WORD_LENGTH = max(len(word) for word in WORD_LIST) if WORD_LIST else 0

# Build buckets in one pass (still sorted, but without MAX_WORD_LENGTH full scans)
_tmp: list[list[str]] = [[] for _ in range(MAX_WORD_LENGTH + 1)]
for w in WORD_LIST:
    _tmp[len(w)].append(w)

N_RAREST_LETTERS = 5
_LETTER_COUNTS = SortedList(
    [
        (ch, sum(1 for word in WORD_LIST if ch in word))
        for ch in (chr(i) for i in range(ord("A"), ord("Z") + 1))
    ],
    key=lambda item: item[1],
)
RARE_LETTERS = {ch for ch, _ in _LETTER_COUNTS[:N_RAREST_LETTERS]}

print(f"Identified rare letters: {', '.join(sorted(RARE_LETTERS))}")


def word_sort_key(word: str) -> tuple[int, str]:
    """Key function for sorting words within each length bucket.

    Awards words with more rare letters a higher priority, as these will be more useful in
    restricting the domains of intersecting slots on the board.
    """
    rare_count = sum(1 for ch in word if ch in RARE_LETTERS)
    return (-rare_count, word)


WORDS_BY_LENGTH = [SortedList(bucket, key=word_sort_key) for bucket in _tmp]
del _tmp

WORD_INDEXES = {}
"""Maps words to their indexes in `WORDS_BY_LENGTH[len(word)]`.

Note words of different lengths may have the same index in their respective lists.
"""
for length in range(1, MAX_WORD_LENGTH + 1):
    WORD_INDEXES.update({w: index for index, w in enumerate(WORDS_BY_LENGTH[length])})

WordBuckets = list[list[list[bitarray]]]
"""A list of WordBits.

Element [length][pos][ch] is a bit array mapping words of length `length` with character `ch`
at position `pos` to True, and those without to False.
"""


def create_word_buckets() -> WordBuckets:
    """Create word buckets for fast lookup.

    Element [length][pos][ch] is a WordBits mapping words of length `length` with character `ch`
    at position `pos` to True, and those without to False.
    """
    if MAX_WORD_LENGTH == 0:
        return []

    buckets: WordBuckets = [
        [
            [
                zeros(len(WORDS_BY_LENGTH[length]))
                for _ in range(26)  # Characters A-Z
            ]
            for _ in range(length)  # Positions in the word
        ]
        for length in range(MAX_WORD_LENGTH + 1)  # By word length
    ]

    # Set the appropriate entries to True
    # It is very important that WORDS_BY_LENGTH[length] is both sorted and never modified,
    # so that all bitarrays align correctly.
    for length in range(1, MAX_WORD_LENGTH + 1):
        for word_index, word in enumerate(WORDS_BY_LENGTH[length]):
            for pos, ch in enumerate(word):
                ch_index = ord(ch) - ord("A")
                buckets[length][pos][ch_index][word_index] = True

    print("Created word buckets for fast lookup.")
    return buckets


WORD_BUCKETS = create_word_buckets()
"""Precomputed word buckets for fast lookup."""


def get_words(pattern: str | bytearray) -> list[str]:
    """Get all words matching the given pattern.

    Args:
        pattern: A string or bytearray representing the pattern to match.
            The pattern may include '.' as a wildcard character.

    Returns:
        A list of words matching the pattern.
    """
    length = len(pattern)
    if length > MAX_WORD_LENGTH:
        return []

    # Get the intersection of WordBits for the pattern
    bits = ones(len(WORDS_BY_LENGTH[length]))
    pattern = bytearray(pattern, "utf-8") if isinstance(pattern, str) else pattern
    for pos, ch in enumerate(pattern):
        if ch == ord("."):
            continue
        ch_index = ch - ord("A")
        if not (0 <= ch_index < 26):
            raise ValueError(f"Invalid character '{chr(ch)}' in pattern.")
        char_bits = WORD_BUCKETS[length][pos][ch_index]
        bits &= char_bits
        if not bits.any():
            return []

    # Collect matching words
    return [WORDS_BY_LENGTH[length][i] for i in bits.search(1)]


if __name__ == "__main__":
    print(f"Word list file found at: {WORDLIST_PATH}")
    print(f"Total words loaded: {len(WORD_LIST)}")
    print(f"Maximum word length: {MAX_WORD_LENGTH}")

    test_pattern = ".OU."
    test_words = get_words(test_pattern)
    print(test_words)
    assert "YOUR" in test_words
    assert "GOUT" in test_words
    assert "BOOM" not in test_words  # third letter is not 'U'
