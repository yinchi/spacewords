"""Utility functions for the Spacewords solver."""

from collections import Counter
from collections.abc import Collection, Mapping
from enum import IntEnum
from functools import lru_cache
from typing import TypeAlias

from sortedcontainers import SortedSet

from spacewords.board import Board
from spacewords.solver.config import config as solver_config

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S.%f %Z%z"


class Direction(IntEnum):
    """Enumeration for word slot directions."""

    ACROSS = 0
    DOWN = 1


Slot: TypeAlias = tuple[Direction, int, int]
Domain: TypeAlias = Collection[str]


def is_playable(to_play: Counter[str], tiles: Counter[str]) -> bool:
    """Returns whether the tiles to play can be formed from the available tiles.

    Args:
        to_play (Counter[str]): A counter of the tiles needed to play.
        tiles (Counter[str]): A counter of the available tiles.
    """
    return all(to_play[ch] <= tiles[ch] for ch in to_play)


@lru_cache(maxsize=300_000)
def get_word_counter(word: str) -> Counter[str]:
    """Return a cached Counter for a word.

    This is a major hotspot in constraint propagation and slot-domain construction.

    Note: the returned Counter must be treated as immutable.
    """
    return Counter(word)


def get_word_positions(board: Board) -> list[tuple[Direction, int, int]]:
    """Get the positions of all word slots on the board.

    Args:
        board (Board): The puzzle board as a Board object.

    Returns:
        A list of tuples representing word positions. Each tuple is of the form
        (direction, start_idx, length).
    """
    word_positions: list[tuple[Direction, int, int]] = []

    # Find across words
    for row in range(board.n_rows):  # Iterate over rows
        for left in range(board.n_cols):  # Iterate over columns
            # Check if the cell [row, left] is the start of a word
            if board[row, left] != "#" and (left == 0 or board[row, left - 1] == "#"):
                # Find the end of the word
                right = left
                while right < board.n_cols and board[row, right] != "#":
                    right += 1

                # Only consider words of length > 1 (should always be true for a connected board)
                length = right - left
                if length > 1:
                    # Add the word positions to the dictionary
                    start_idx = board.get_1d_idx(row, left)
                    word_positions.append((Direction.ACROSS, start_idx, length))

    # Find down words
    for col in range(board.n_cols):
        for top in range(board.n_rows):
            # Check if the cell [top, col] is the start of a word
            if board[top, col] != "#" and (top == 0 or board[top - 1, col] == "#"):
                # Find the end of the word
                bottom = top
                while bottom < board.n_rows and board[bottom, col] != "#":
                    bottom += 1

                # Only consider words of length > 1 (should always be true for a connected board)
                length = bottom - top
                if length > 1:
                    # Add the word positions to the dictionary
                    start_idx = board.get_1d_idx(top, col)
                    word_positions.append((Direction.DOWN, start_idx, length))

    return word_positions


def get_slot_indices(board: Board) -> dict[tuple[Direction, int, int], list[int]]:
    """Get the mapping of slots to their cell indices.

    Args:
        board (Board): The puzzle board as a Board object.

    Returns:
        A dictionary mapping (direction, start_idx, length) to a list of cell indices for each
        slot.  The indexes should be the result of calling `get_word_positions`.
    """
    slot_indices: dict[tuple[Direction, int, int], list[int]] = {}

    word_positions = get_word_positions(board)

    for direction, start_idx, length in word_positions:
        indices: list[int] = []
        if direction == Direction.ACROSS:
            indices = list(board.across_range(start_idx, length))
        elif direction == Direction.DOWN:
            indices = list(board.down_range(start_idx, length))
        else:
            raise ValueError(f"Invalid direction: {direction}")

        slot_indices[(direction, start_idx, length)] = indices

    return slot_indices


def get_slot_words(
    board: Board,
    slot_indices: dict[Slot, list[int]],
    words: set[str],
    word_map: dict[tuple[int, int, str], set[str]],
    available_tiles: Counter[str],
    *,
    words_by_length: Mapping[int, set[str]] | None = None,
) -> dict[Slot, Domain]:
    """Get possible words for each slot on the board.

    A slot is defined by its direction, starting index (row or column), and length
    and maps to the list of (1D) cell indices that make up that word.

    Args:
        board (Board): The puzzle board as a Board object.
        slot_indices (dict): Mapping of (direction, start_idx, length) to list of cell indices.
        words (set[str]): Set of valid words.
        word_map (dict): Mapping of (word_length, position_in_word, character) to sets of words.
        available_tiles (Counter[str]): Counter of available tiles (excluding played tiles).
        words_by_length (Mapping[int, set[str]] | None): Optional pre-bucketed words keyed by
            word length to avoid repeated scanning.

    Returns:
        A dictionary mapping (direction, start_idx, length) to sets of possible words for each slot.
    """
    slot_words: dict[Slot, Domain] = {}
    empty: set[str] = set()

    for (dir, start_idx, length), indices in slot_indices.items():
        keys: list[tuple[int, int, str]] = []
        # For each cell in the slot, remove words that don't match the board character
        for word_pos, board_idx in enumerate(indices):
            board_char = board.data[board_idx]
            if board_char not in (".", "#"):
                key = (length, word_pos, board_char)
                keys.append(key)
        # Always restrict by slot length, even when there are no fixed-letter constraints.
        if words_by_length is not None:
            length_words = words_by_length.get(length, empty)
        else:
            length_words = {w for w in words if len(w) == length}
        possible_words = set.intersection(
            *(word_map.get(key, empty) for key in keys),
            length_words,
        )

        # Further filter words based on available tiles
        filled_letters = Counter(
            board.data[idx] for idx in indices if board.data[idx] not in (".", "#")
        )

        def can_form_word(word: str) -> bool:
            needed_tiles = get_word_counter(word) - filled_letters
            return is_playable(needed_tiles, available_tiles)

        possible_words = {w for w in possible_words if can_form_word(w)}
        possible_words_domain: Domain = (
            SortedSet(possible_words) if solver_config.deterministic else possible_words
        )

        # Store the computed list of possible words for the slot
        slot_words[(dir, start_idx, length)] = possible_words_domain

    return slot_words


def validate_board(board: Board, word_list: set[str]) -> bool:
    """Validate that the board is solved: no empty "." cells and all words are valid."""
    # Get the slot indices (defined by direction, start index, length) for the board
    slot_idxs = get_slot_indices(board)

    for indices in slot_idxs.values():
        # Extract the word from the board
        word = "".join(board[idx] for idx in indices)

        # Check for empty cells or invalid words
        if "." in word or word not in word_list:
            return False

    # If all words are valid and no empty cells, the board is valid
    return True
