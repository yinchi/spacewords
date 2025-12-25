"""Module defining the Slot class representing a slot on the board."""

import re
from enum import IntEnum
from typing import NamedTuple

from bitarray import bitarray
from bitarray.util import ones

from spacewords2.words import WORD_BUCKETS, WORDS_BY_LENGTH

VALID_SLOT_PATTERN = re.compile(r"^[A-Z\.]+$")
"""Regex pattern for validating slot patterns.

A valid slot pattern consists of uppercase letters (A-Z), and '.' for empty cells.
"""


class Direction(IntEnum):
    """Enumeration for slot directions."""

    ACROSS = 0
    DOWN = 1


class SlotPosition(NamedTuple):
    """Datatype representing the position of a slot."""

    direction: Direction
    start_row: int
    start_col: int


class Slot:
    """Represents a single slot on the board.

    A slot represents a sequence of cells on the board where a word can be placed.
    """

    def __init__(self, slot_start: SlotPosition, length: int, pattern: str | None = None):
        # Check input validity
        if length <= 0:
            raise ValueError("Slot length must be positive.")
        if pattern is None:
            pattern = "." * length
        if len(pattern) != length:
            raise ValueError("Slot pattern length must match slot length.")
        if not VALID_SLOT_PATTERN.match(pattern):
            raise ValueError("Slot pattern contains invalid characters.")

        self.pos: SlotPosition | None = slot_start
        """Starting position of the slot, as (direction, start_row, start_col)."""

        self.length: int = length
        """Length of the slot."""

        self.intersections: list[Intersection | None] = [None] * length
        """List of intersections with other slots.

        If `self.intersections[i]` is `None`, there is no intersection at cell `i`.
        Otherwise, it is a tuple (`other_slot`, `j`) indicating that
        the `i`th cell of this slot is also the `j`th cell of `other_slot`.
        """

        self.domain: bitarray = ones(len(WORDS_BY_LENGTH[length]))
        """Bitarray representing the set of possible words that can fit in this slot.

        The `i`th bit is True if the WORDS_BY_LENGTH[length][i] is a possible word for this slot.
        This may be restricted by intersections with other slots.
        """

        # Restrict self.domain based on initial pattern
        for pos, ch in enumerate(pattern):
            if ch == ".":
                continue
            if "A" <= ch <= "Z":
                ch_index = ord(ch) - ord("A")
                char_bits = WORD_BUCKETS[length][pos][ch_index]
                self.domain &= char_bits
            else:
                raise ValueError(f"Invalid character '{ch}' in slot pattern.")

        self.domain_size: int = self.domain.count()
        """Number of possible words for this slot."""

    def get_words(self) -> list[str]:
        """Get all words matching the current slot pattern."""
        return [WORDS_BY_LENGTH[self.length][i] for i in self.domain.search(1)]

    @property
    def fixed(self) -> bool:
        """Check if the slot has a fixed word (only one possible word)."""
        return self.domain_size == 1

    @property
    def fixed_word(self) -> str | None:
        """Get the fixed word if the slot has only one possible word, else None."""
        if not self.fixed:
            return None
        index = self.domain.find(1)
        # Disable assertion in production for efficiency
        assert index != -1, "Domain size is 1 but no bit set in domain."
        return WORDS_BY_LENGTH[self.length][index]


class Intersection(NamedTuple):
    """Represents an intersection between two slots."""

    other_slot: Slot
    index_in_other_slot: int
