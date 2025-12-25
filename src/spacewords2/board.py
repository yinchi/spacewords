"""Board representation for the Spacewords game."""

import re
from collections import defaultdict
from enum import IntEnum
from typing import NamedTuple

from bitarray import bitarray
from bitarray.util import ones

from spacewords2.tiles import TileBag, create_tile_bag
from spacewords2.words import WORD_BUCKETS, WORD_INDEXES, WORDS_BY_LENGTH

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


class _UndoInfo(NamedTuple):
    """Marker into the undo trails for a single move."""

    layout_mark: int
    tile_mark: int
    domain_mark: int


class Board:
    """Represents the game board."""

    def __init__(self, layout: str, n_rows: int, n_cols: int):
        def _idx(row: int, col: int) -> int:
            return row * n_cols + col

        self.layout: bytearray = bytearray(layout, "utf-8")
        """String representation of the board layout.

        Cells are listed row-wise. '#' represents black cells, '.' represents empty cells.
        Cells with letters A-Z represent fixed letters on the board.
        """

        self.n_rows: int = n_rows
        """Number of rows in the board."""

        self.n_cols: int = n_cols
        """Number of columns in the board."""

        self.slot_map: dict[SlotPosition, Slot] = {}
        """Mapping from positions to Slot objects."""

        # Undo trails (used by place_word / undo_place_word)
        self._layout_changes: list[tuple[int, int]] = []
        self._tile_decrements: list[int] = []
        self._domain_changes: list[tuple[SlotPosition, bitarray, int]] = []

        self.intersection_map: defaultdict[tuple[int, int], list[Intersection]] = defaultdict(list)
        """Mapping from cell positions (row, col) to a list of
        (slot_start, index_in_slot) entries."""

        # Process rows for ACROSS slots
        for row in range(n_rows):
            left = 0
            while left < n_cols:
                pattern = []
                # Get all characters until a '#'
                right = left
                while right < n_cols and layout[idx := _idx(row, right)] != "#":
                    pattern.append(layout[idx])
                    right += 1
                # Create slot if length > 1
                length = right - left
                if length > 1:
                    slot_start = SlotPosition(Direction.ACROSS, row, left)
                    slot = Slot(slot_start, length, "".join(pattern))
                    self.slot_map[slot_start] = slot
                    for i in range(length):
                        self.intersection_map[(row, left + i)].append(Intersection(slot, i))
                # Move past the '#' to the next potential slot
                left = right + 1

        # Process columns for DOWN slots
        for col in range(n_cols):
            top = 0
            while top < n_rows:
                pattern = []
                # Get all characters until a '#'
                bottom = top
                while bottom < n_rows and layout[idx := _idx(bottom, col)] != "#":
                    pattern.append(layout[idx])
                    bottom += 1
                # Create slot if length > 1
                length = bottom - top
                if length > 1:
                    slot_start = SlotPosition(Direction.DOWN, top, col)
                    slot = Slot(slot_start, length, "".join(pattern))
                    self.slot_map[slot_start] = slot
                    for i in range(length):
                        self.intersection_map[(top + i, col)].append(Intersection(slot, i))
                # Move past the '#' to the next potential slot
                top = bottom + 1

        # For each slot, find intersections with other slots
        for slot in self.slot_map.values():
            if not slot.pos:
                raise RuntimeError("Uninitialized slot_start.")
            direction, start_row, start_col = slot.pos
            other_direction = Direction.DOWN if direction == Direction.ACROSS else Direction.ACROSS
            for i in range(slot.length):
                if direction == Direction.ACROSS:
                    cell = (start_row, start_col + i)
                else:
                    cell = (start_row + i, start_col)
                for intersection in self.intersection_map[cell]:
                    if not intersection.other_slot.pos:
                        raise RuntimeError("Uninitialized position in other_slot.")
                    if intersection.other_slot.pos.direction == other_direction:
                        slot.intersections[i] = intersection

    def print(self):
        """Prints the board layout."""
        for row in range(self.n_rows):
            line = (
                self.layout[row * self.n_cols : (row + 1) * self.n_cols]
                .decode("utf-8")
                .replace("#", "█")
            )
            print(line)

    def place_word(self, pos: SlotPosition, word: str, tile_bag: TileBag) -> _UndoInfo:
        """Place a word in the specified slot.  Modifies both the board and `tile_bag` in-place.

        Args:
            pos: The position of the slot where the word will be placed.
            word: The word to place in the slot.
            tile_bag: The current tile bag representing available tiles.  Modified in-place.

        Returns:
            An _UndoInfo object that can be used to undo the word placement.

        Raises:
            ValueError: If the word cannot be placed due to conflicts or invalidity.
        """
        word = word.upper()

        # Retrieve the slot
        slot = self.slot_map.get(pos)
        if slot is None:
            raise ValueError(f"No slot found at position {pos}.")
        if slot.pos is None:
            raise RuntimeError("Uninitialized position in slot.")

        # Check word length
        if len(word) != slot.length:
            raise ValueError("Word length does not match slot length.")

        # Check word validity
        if word not in WORD_INDEXES:
            raise ValueError(f"Word '{word}' is not in the word list.")

        undo_info = _UndoInfo(
            layout_mark=len(self._layout_changes),
            tile_mark=len(self._tile_decrements),
            domain_mark=len(self._domain_changes),
        )

        changed_domains: set[SlotPosition] = set()

        try:
            # Update board layout, validating against existing letters and tracking tiles used
            for i in range(slot.length):
                direction, start_row, start_col = slot.pos
                if direction == Direction.ACROSS:
                    row, col = start_row, start_col + i
                else:
                    row, col = start_row + i, start_col
                idx = row * self.n_cols + col
                board_ch = self.layout[idx]
                word_ch = ord(word[i])
                if board_ch == ord("."):
                    if tile_bag[word_ch - ord("A")] > 0:
                        tile_idx = word_ch - ord("A")
                        tile_bag[tile_idx] -= 1
                        self._tile_decrements.append(tile_idx)
                    else:
                        raise ValueError(f"Not enough tiles to place word '{word}' at {pos}.")
                    self._layout_changes.append((idx, board_ch))
                    self.layout[idx] = word_ch
                elif board_ch != word_ch:
                    raise ValueError(
                        f"Conflict placing word '{word}' at {pos}: "
                        f"board has '{chr(board_ch)}' but word has '{chr(word_ch)}' at "
                        f"position {i}."
                    )

            # The first slot we change is the one we are placing the word in.
            if pos not in changed_domains:
                self._domain_changes.append((pos, slot.domain, slot.domain_size))
                changed_domains.add(pos)

            # Place the word by setting its slot's domain to a single word (by swapping
            # the domain reference, not mutating the existing bitarray).
            new_domain = bitarray(len(WORDS_BY_LENGTH[slot.length]))
            new_domain.setall(False)
            new_domain[WORD_INDEXES[word]] = True
            slot.domain = new_domain
            slot.domain_size = 1

            # Update domains of intersecting slots
            for i, intersection in enumerate(slot.intersections):
                if intersection is None:
                    continue
                other_slot = intersection.other_slot
                j = intersection.index_in_other_slot
                ch = word[i]
                ch_index = ord(ch) - ord("A")
                char_bits = WORD_BUCKETS[other_slot.length][j][ch_index]

                if other_slot.pos is None:
                    raise RuntimeError("Uninitialized slot_start in other_slot.")

                other_slot_start = other_slot.pos
                if other_slot_start not in changed_domains:
                    # Only save old domain once (it may be updated multiple times)
                    self._domain_changes.append(
                        (other_slot_start, other_slot.domain, other_slot.domain_size)
                    )
                    changed_domains.add(other_slot_start)

                # Swap domain reference instead of mutating in place.
                new_other_domain = other_slot.domain & char_bits
                if not new_other_domain.any():
                    raise ValueError(
                        f"Placing word '{word}' causes slot at {other_slot_start} to have "
                        "no valid words."
                    )
                other_slot.domain = new_other_domain
                other_slot.domain_size = new_other_domain.count()

            return undo_info

        except Exception:
            # On error, restore previous state
            self.undo_place_word(undo_info, tile_bag)
            raise

    def undo_place_word(self, undo_info: _UndoInfo, tile_bag: TileBag) -> None:
        """Undo a previously placed word on the board.

        Args:
            undo_info: The information needed to undo the word placement.
            tile_bag: The current tile bag representing available tiles.  Restored in-place.
        """
        # Restore slot domains (and sizes)
        while len(self._domain_changes) > undo_info.domain_mark:
            pos, old_domain, old_size = self._domain_changes.pop()
            slot = self.slot_map.get(pos)
            if slot is None:
                raise RuntimeError(f"No slot found at position {pos} during undo.")
            slot.domain = old_domain
            slot.domain_size = old_size

        # Restore board layout
        while len(self._layout_changes) > undo_info.layout_mark:
            idx, old_byte = self._layout_changes.pop()
            self.layout[idx] = old_byte

        # Restore tile bag
        while len(self._tile_decrements) > undo_info.tile_mark:
            tile_idx = self._tile_decrements.pop()
            tile_bag[tile_idx] += 1

    def solve(self, tile_bag: TileBag, first_pos: SlotPosition) -> "Board":
        """Solve the board using the available tiles in the tile bag (TODO).

        Attempts to fill in all slots on the board using words from the word list, subject
        to the constraints of the current board layout and the available tiles.
        Uses AC3 to reduce domains and backtracking to find a solution.

        When backtracking, slots to attempt to fill are chosen in order of:

        1. Manhattan distance from `first_pos`.  This "flood-fills" out from the first slot and
           increases the chance of domain reductions from intersections.  Ensures the slot at
           `first_pos` is attempted first.
        2. Fewest remaining possible words in the slot's domain (MRV heuristic).
        3. Largest number of intersections with other slots (degree heuristic).
        4. Across slots before down slots.
        5. Top to bottom, left to right.

        Furthermore, within each slot, words to attempt to place are chosen in order of
        appearance in `WORDS_BY_LENGTH[slot.length]` (filtered by `slot.domain`).  This is dictated
        by the sorting function `word_sort_key`, which prioritizes words with rare letters first.

        Args:
            tile_bag: The current tile bag representing available tiles.
            first_pos: Position of the starting slot to begin solving from.
                Used to distribute solving attempts in a multi-threaded context.

        Returns:
            A new Board instance representing the solved board.

        Raises:
            ValueError: If the board cannot be solved with the given tiles.
        """
        # Check that first_pos is valid
        if first_pos not in self.slot_map:
            raise ValueError(f"first_pos {first_pos} is not a valid slot position on the board.")

        def _manhattan_distance(pos: SlotPosition) -> int:
            """Get the Manhattan distance from `first_pos` to `pos`."""
            r1, c1 = pos.start_row, pos.start_col
            r2, c2 = first_pos.start_row, first_pos.start_col
            return abs(r1 - r2) + abs(c1 - c2)

        slot_degrees: dict[SlotPosition, int] = {
            slot_start: sum(1 for inter in slot.intersections if inter is not None)
            for slot_start, slot in self.slot_map.items()
        }

        manhattan_distances = {pos: _manhattan_distance(pos) for pos in self.slot_map.keys()}

        def _slot_sort_key(pos: SlotPosition) -> tuple[int, int, int, int, int]:
            """Key function for sorting slots to fill."""
            slot = self.slot_map.get(pos)
            if slot is None:
                raise RuntimeError(f"No slot found at position {pos} during sorting.")
            return (
                manhattan_distances[pos],
                slot.domain_size,
                -slot_degrees[pos],
                pos.direction.value,  # 0 for ACROSS, 1 for DOWN (to prefer ACROSS)
                pos.start_row * self.n_cols + pos.start_col,
            )

        raise NotImplementedError("Board solving not yet implemented.")  # TODO


def test_board():
    """Check the following test board.

    Board layout:

    #B.#
    LOOK
    #O.#
    """
    layout = "".join(
        [
            "#B.#",
            "LOOK",
            "#O.#",
        ]
    )
    board = Board(layout, 3, 4)
    assert len(board.slot_map) == 5  # Expected number of slots (3 across, 2 down)

    # Check slots with fixed words
    assert board.slot_map[SlotPosition(Direction.ACROSS, 1, 0)].get_words() == ["LOOK"]
    assert board.slot_map[SlotPosition(Direction.DOWN, 0, 1)].get_words() == ["BOO"]

    # Intersections at both 'O's in "LOOK"
    assert board.slot_map[SlotPosition(Direction.ACROSS, 1, 0)].intersections == [
        None,
        Intersection(board.slot_map[SlotPosition(Direction.DOWN, 0, 1)], 1),
        Intersection(board.slot_map[SlotPosition(Direction.DOWN, 0, 2)], 1),
        None,
    ]

    # Intersections with column "BOO"
    assert board.slot_map[SlotPosition(Direction.DOWN, 0, 1)].intersections == [
        Intersection(board.slot_map[SlotPosition(Direction.ACROSS, 0, 1)], 0),
        Intersection(board.slot_map[SlotPosition(Direction.ACROSS, 1, 0)], 1),
        Intersection(board.slot_map[SlotPosition(Direction.ACROSS, 2, 1)], 0),
    ]

    # Create a tile bag with one of each letter A-Z
    tile_bag = create_tile_bag("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Try to place BOX in third column (column 2), fails since BB is not in word list
    # Automatically undos on failure
    try:
        undo_info = board.place_word(SlotPosition(Direction.DOWN, 0, 2), "BOX", tile_bag)
        assert False, "Expected ValueError when placing invalid word."
    except ValueError:
        pass

    # What are the possible words for column 2, ignoring intersections?
    words = board.slot_map[SlotPosition(Direction.DOWN, 0, 2)].get_words()

    words_that_fit = []
    for word in words:
        try:
            # If word fits, append to list and undo
            undo_info = board.place_word(SlotPosition(Direction.DOWN, 0, 2), word, tile_bag)
            words_that_fit.append(word)
            board.undo_place_word(undo_info, tile_bag)
        except ValueError:
            # `place_word` automatically undos on failure
            continue

    print(f"Valid words for column 2 (accounting for intersections):\n{words_that_fit}")
    assert "YOU" in words_that_fit

    undo_info = board.place_word(SlotPosition(Direction.DOWN, 0, 2), "YOU", tile_bag)
    board.print()  # Print the board after placing YOU

    # Placed U and Y, used existing O on board
    tile_bag_for_comparison = create_tile_bag("ABCDEFGHIJKLMONPQRSTVWXZ")
    assert tile_bag == tile_bag_for_comparison


if __name__ == "__main__":
    test_board()
