"""Board representation for the Spacewords game."""

import random
import sys
from array import array
from collections import defaultdict, deque
from dataclasses import dataclass
from time import time
from typing import NamedTuple

from bitarray import bitarray
from bitarray.util import count_and, zeros

from spacewords2.slot import Direction, Intersection, Slot, SlotPosition
from spacewords2.tiles import TileBag, create_tile_bag, tile_bag_to_string
from spacewords2.util import int_comma, time_str
from spacewords2.words import LETTER_RARITY_WEIGHT, WORD_BUCKETS, WORD_INDEXES, WORDS_BY_LENGTH

REPORT_INTERVAL = 100_000
"""Interval (in number of boards checked) for reporting progress during solving."""

REPORT_HI_SCORE_MIN = 1
"""Report new best board only if at least this many slots are filled."""

TILE_AWARE_MRV_CANDIDATES = 8
"""Number of top MRV candidates to consider for tile-aware selection."""

TILE_AWARE_MRV_MAX_CANDIDATES = 32
"""Maximum number of candidates to consider for tile-aware MRV (caps overhead)."""

RANDOMIZE_TIES = False
"""Whether to randomize slot-selection ties during solving."""


class _UndoInfo(NamedTuple):
    """Marker into the undo trails for a single move.

    Undo will reverse changes until the remaining lengths of the trails match these marks.
    """

    layout_mark: int
    tile_mark: int
    domain_mark: int
    filled_mark: int


class _UndoTrails(NamedTuple):
    """Trails for undoing changes to the board and tile bag."""

    layout_changes: list[tuple[int, int]]
    """List of (index, old_byte) tuples representing changes to the board layout."""

    tile_decrements: array[int]
    """List of tile indices that were decremented in the tile bag."""

    domain_changes: list[tuple[SlotPosition, bitarray, int]]
    """List of (slot_position, old_domain, old_domain_size) tuples.

    Each tuple represents changes to slot domains.
    """

    filled_changes: list[tuple[SlotPosition, bool]]
    """List of (slot_position, was_unfilled) tuples.

    Used to restore `Board._unfilled_slots` during undo.
    """


NeighborMap = dict[SlotPosition, list[tuple[SlotPosition, int, int]]]
"""Mapping from slot positions to lists of tuples representing intersections.

Each tuple is of the form (other_slot_position, index_in_this_slot, index_in_other_slot).
"""

ArcMap = dict[tuple[SlotPosition, SlotPosition], tuple[int, int]]
"""Mapping from pairs of slot positions to their intersection indices.

Each key is a tuple (slot_position_1, slot_position_2), and the value is a tuple
(index_in_slot_1, index_in_slot_2).
"""


class ConstraintGraph(NamedTuple):
    """Constraint graph representing slot intersections."""

    neighbors: NeighborMap
    """Neighbor mapping from slot positions to their intersecting slots."""

    arc_map: ArcMap
    """Mapping from pairs of slot positions to their intersection indices."""


@dataclass
class SolverStats:
    """Statistics collected during solving."""

    boards_checked: int = 0
    """Number of board states checked during solving."""

    start_time: float = time()
    """Timestamp when solving started."""

    max_depth_reached: int = 0
    """Maximum recursion depth reached during solving."""

    max_filled_slots: int = 0
    """Maximum number of filled slots reached during solving."""


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
        self._undo_trails: _UndoTrails = _UndoTrails([], array("I"), [], [])

        # Cached board indices for each slot (hot-path optimization).
        self._slot_cell_idxs: dict[SlotPosition, tuple[int, ...]] = {}

        # Cached slot-relative indices which are not intersections (static).
        # Used by heuristics to prefer placing rare letters in low-constraint cells.
        self._slot_non_intersect_idxs: dict[SlotPosition, tuple[int, ...]] = {}

        # Tracks which slots are not yet fully filled on the board (contain at least one '.')
        self._unfilled_slots: set[SlotPosition] = set()

        self.intersection_map: defaultdict[tuple[int, int], list[Intersection]] = defaultdict(list)
        """Mapping from cell positions (row, col) to a list of
        (slot_start, index_in_slot) entries."""

        self.solve_stats: SolverStats = SolverStats()
        """Statistics collected during solving."""

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

        # Cache non-intersection positions (slot-relative indices)
        for pos, slot in self.slot_map.items():
            self._slot_non_intersect_idxs[pos] = tuple(
                i for i, inter in enumerate(slot.intersections) if inter is None
            )

        self.constraint_graph = self._build_constraint_graph()
        """Precomputed constraint graph for the board.

        Since board topology does not change, this is built once at initialization.
        """

        # Cache the board indices for each slot (used by place_word / tile checks / etc.).
        for pos, slot in self.slot_map.items():
            if slot.pos is None:
                raise RuntimeError("Uninitialized position in slot.")
            direction, start_row, start_col = slot.pos
            base = start_row * self.n_cols + start_col
            if direction == Direction.ACROSS:
                self._slot_cell_idxs[pos] = tuple(range(base, base + slot.length))
            else:
                self._slot_cell_idxs[pos] = tuple(
                    range(base, base + self.n_cols * slot.length, self.n_cols)
                )

        # Initialize unfilled slots from the starting layout.
        for pos in self.slot_map:
            if any(self.layout[idx] == ord(".") for idx in self._slot_cell_idxs[pos]):
                self._unfilled_slots.add(pos)

    def print(self):
        """Prints the board layout."""
        for row in range(self.n_rows):
            line = (
                self.layout[row * self.n_cols : (row + 1) * self.n_cols]
                .decode("utf-8")
                .replace("#", "█")
            )
            print(line)

    def _build_constraint_graph(
        self,
    ) -> ConstraintGraph:
        """Build the slot-intersection constraint graph.

        Returns:
            (neighbors, arc_map)

            - neighbors[x] is a list of (y, i, j) meaning slot x at index i intersects
              slot y at index j.
            - arc_map[(x, y)] == (i, j) for fast lookup during arc-consistency (AC-3).

        Notes:
            In this puzzle, a pair of perpendicular slots can intersect at most once, so
            (x, y) identifies a unique (i, j).
        """
        neighbors: dict[SlotPosition, list[tuple[SlotPosition, int, int]]] = {
            pos: [] for pos in self.slot_map
        }
        arc_map: dict[tuple[SlotPosition, SlotPosition], tuple[int, int]] = {}

        for pos, slot in self.slot_map.items():
            for i, intersection in enumerate(slot.intersections):
                if intersection is None:
                    continue
                other_slot = intersection.other_slot
                if other_slot.pos is None:
                    raise RuntimeError("Uninitialized position in other_slot.")
                other_pos = other_slot.pos
                j = intersection.index_in_other_slot

                neighbors[pos].append((other_pos, i, j))

                key = (pos, other_pos)
                prev = arc_map.get(key)
                if prev is not None and prev != (i, j):
                    raise RuntimeError(
                        f"Multiple intersections found between {pos} and {other_pos}."
                    )
                arc_map[key] = (i, j)

        return ConstraintGraph(neighbors, arc_map)

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
            layout_mark=len(self._undo_trails.layout_changes),
            tile_mark=len(self._undo_trails.tile_decrements),
            domain_mark=len(self._undo_trails.domain_changes),
            filled_mark=len(self._undo_trails.filled_changes),
        )

        changed_domains: set[SlotPosition] = set()

        try:
            # Cheap pre-check: validate conflicts and ensure the tile bag can cover all new
            # letters needed for this placement, without mutating anything.
            pending_writes: list[tuple[int, int]] = []
            needed_tiles = [0] * 26

            cell_idxs = self._slot_cell_idxs[pos]
            for i, idx in enumerate(cell_idxs):
                board_ch = self.layout[idx]
                word_ch = ord(word[i])

                if board_ch == ord("."):
                    tile_idx = word_ch - ord("A")
                    needed_tiles[tile_idx] += 1
                    pending_writes.append((idx, word_ch))
                elif board_ch != word_ch:
                    raise ValueError(
                        f"Conflict placing word '{word}' at {pos}: "
                        f"board has '{chr(board_ch)}' but word has '{chr(word_ch)}' at "
                        f"position {i}."
                    )

            for tile_idx, count in enumerate(needed_tiles):
                if count and tile_bag[tile_idx] < count:
                    raise ValueError(f"Not enough tiles to place word '{word}' at {pos}.")

            # Apply board layout changes + tile decrements (now guaranteed to succeed).
            for idx, word_ch in pending_writes:
                tile_idx = word_ch - ord("A")
                tile_bag[tile_idx] -= 1
                self._undo_trails.tile_decrements.append(tile_idx)
                self._undo_trails.layout_changes.append((idx, self.layout[idx]))
                self.layout[idx] = word_ch

            # Update unfilled-slot tracking. Only slots that include newly filled '.' cells
            # can transition from unfilled -> filled.
            affected_slots: set[SlotPosition] = {pos}
            # Build a quick lookup of written indices for this placement.
            written_idxs = {idx for idx, _ in pending_writes}
            for i, idx in enumerate(self._slot_cell_idxs[pos]):
                if idx not in written_idxs:
                    continue
                intersection = slot.intersections[i]
                if intersection is None:
                    continue
                other_pos = intersection.other_slot.pos
                if other_pos is None:
                    raise RuntimeError("Uninitialized slot_start in other_slot.")
                affected_slots.add(other_pos)

            for affected_pos in affected_slots:
                if affected_pos not in self._unfilled_slots:
                    continue
                affected_slot = self.slot_map[affected_pos]
                if affected_slot.pos is None:
                    raise RuntimeError("Uninitialized position in slot.")
                if all(
                    self.layout[affected_idx] != ord(".")
                    for affected_idx in self._slot_cell_idxs[affected_pos]
                ):
                    self._undo_trails.filled_changes.append((affected_pos, True))
                    self._unfilled_slots.remove(affected_pos)

            # The first slot we change is the one we are placing the word in.
            if pos not in changed_domains:
                self._undo_trails.domain_changes.append((pos, slot.domain, slot.domain_size))
                changed_domains.add(pos)

            # Place the word by setting its slot's domain to a single word (by swapping
            # the domain reference, not mutating the existing bitarray).
            new_domain = zeros(len(WORDS_BY_LENGTH[slot.length]))
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
                    self._undo_trails.domain_changes.append(
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
        while len(self._undo_trails.domain_changes) > undo_info.domain_mark:
            pos, old_domain, old_size = self._undo_trails.domain_changes.pop()
            slot = self.slot_map.get(pos)
            if slot is None:
                raise RuntimeError(f"No slot found at position {pos} during undo.")
            slot.domain = old_domain
            slot.domain_size = old_size

        # Restore board layout
        while len(self._undo_trails.layout_changes) > undo_info.layout_mark:
            idx, old_byte = self._undo_trails.layout_changes.pop()
            self.layout[idx] = old_byte

        # Restore tile bag
        while len(self._undo_trails.tile_decrements) > undo_info.tile_mark:
            tile_idx = self._undo_trails.tile_decrements.pop()
            tile_bag[tile_idx] += 1

        # Restore unfilled-slot tracking
        while len(self._undo_trails.filled_changes) > undo_info.filled_mark:
            pos, was_unfilled = self._undo_trails.filled_changes.pop()
            if was_unfilled:
                self._unfilled_slots.add(pos)
            else:
                self._unfilled_slots.discard(pos)

    def solve(
        self,
        tile_bag: TileBag,
        first_pos: SlotPosition,
        *,
        randomize_ties: bool = RANDOMIZE_TIES,
        seed: int | None = None,
    ) -> "Board":
        """Solve the board using the available tiles in the tile bag (TODO).

        Attempts to fill in all slots on the board using words from the word list, subject
        to the constraints of the current board layout and the available tiles.
        Uses AC3 to reduce domains and backtracking to find a solution.  Stops at the first
        valid solution found, upon which `self` is the solved board (in-place updates).

        When backtracking, slots to attempt to fill are chosen based on `_slot_sort_key()`
        as defined in `solve_helper()`.

        At each level of recursion, the slot chosen becomes fixed with a word from its domain,
        and is removed from further consideration.  The domains of intersecting slots are updated
        accordingly, and AC-3 is run to further reduce domains.  If any slot ends up with an empty
        domain, the placement is undone and the next word in the domain is attempted.  If all words
        in the domain are exhausted, backtrack to the previous slot.  If all words in the initial
        slot are exhausted, the board is unsolvable with the given tiles and a ValueError is raised.

        Furthermore, within each slot, words to attempt to place are chosen in order of
        appearance in `WORDS_BY_LENGTH[slot.length]` (filtered by `slot.domain`).  This is dictated
        by the sorting function `word_sort_key`, which prioritizes words with rare letters first.

        Args:
            tile_bag: The current tile bag representing available tiles.
            first_pos: Position of the starting slot to begin solving from.
                Used to distribute solving attempts in a multi-threaded context.
            randomize_ties: If True, break slot-selection ties using a stable random value per
                slot (seeded by `seed`) instead of deterministic direction/position ordering.
                This can diversify search and avoid repeatedly exploring the same subtree.
            seed: Optional RNG seed used when `randomize_ties` is True.

        Raises:
            ValueError: If the board cannot be solved with the given tiles.
        """
        slot_tiebreak: dict[SlotPosition, float] | None = None
        if randomize_ties:
            rng = random.Random(seed)
            # Stable per-slot tiebreak values for the entire solve.
            slot_tiebreak = {pos: rng.random() for pos in self.slot_map}

        def _manhattan_distance(pos: SlotPosition) -> int:
            """Get the Manhattan distance from `first_pos` to `pos`."""
            r1, c1 = pos.start_row, pos.start_col
            r2, c2 = first_pos.start_row, first_pos.start_col
            return abs(r1 - r2) if pos.direction == Direction.ACROSS else abs(c1 - c2)

        slot_degrees: dict[SlotPosition, int] = {
            slot_start: slot.length
            # slot_start: sum(1 for inter in slot.intersections if inter is not None)
            for slot_start, slot in self.slot_map.items()
        }

        # Check that first_pos is valid
        if first_pos not in self.slot_map:
            raise ValueError(f"first_pos {first_pos} is not a valid slot position on the board.")

        manhattan_distances = {pos: _manhattan_distance(pos) for pos in self.slot_map.keys()}

        def _prune_domains_by_tiles() -> None:
            """One-off pruning using the full tile bag.

            This removes any word from a slot's domain which would require more tiles of
            some letter than exist in the full tile bag, accounting for already-filled
            letters on the board (which do not consume tiles).

            This pruning is sound because tile availability only decreases during search.
            """
            for pos, slot in self.slot_map.items():
                if slot.pos is None:
                    raise RuntimeError("Uninitialized position in slot.")

                # Effective availability per letter for this slot: tile bag plus letters
                # already present on the board in this slot.
                effective = [int(x) for x in tile_bag]
                for idx in self._slot_cell_idxs[pos]:
                    ch_byte = self.layout[idx]
                    if ord("A") <= ch_byte <= ord("Z"):
                        effective[ch_byte - ord("A")] += 1

                to_clear: array[int] = array("I")  # Indexes of unplacable words
                for word_idx in slot.domain.search(1):
                    w = WORDS_BY_LENGTH[slot.length][word_idx]
                    counts: dict[int, int] = {}
                    ok = True
                    for ch in w:
                        ch_byte = ord(ch) - ord("A")
                        new_count = counts.get(ch_byte, 0) + 1
                        if new_count > effective[ch_byte]:
                            ok = False
                            break
                        counts[ch_byte] = new_count
                    if not ok:
                        to_clear.append(word_idx)

                if to_clear:
                    for word_idx in to_clear:
                        slot.domain[word_idx] = False
                    slot.domain_size -= len(to_clear)

                if slot.domain_size <= 0 or not slot.domain.any():
                    raise ValueError(
                        f"No valid words remain for slot at {pos} under the given tile bag."
                    )

        # One-off pruning before entering recursive solve.
        _prune_domains_by_tiles()

        # Initial filled slots (tracked incrementally from board layout).
        initial_filled = len(self.slot_map) - len(self._unfilled_slots)
        print(f"Initial filled slots: {initial_filled} / {len(self.slot_map)}")

        if not self._unfilled_slots:
            return self

        try:
            self.solve_helper(
                tile_bag,
                manhattan_distances,
                slot_degrees,
                depth=0,
                slot_tiebreak=slot_tiebreak,
                first_pos=first_pos,
            )
            return self
        except ValueError as e:
            # ValueError has propagated up to the top level: no solution found
            print(e, file=sys.stderr)
            raise ValueError("No solution found for the board with the given tile bag.") from e

    def solve_helper(
        self,
        tile_bag: TileBag,
        manhattan_distances: dict[SlotPosition, int],
        slot_degrees: dict[SlotPosition, int],
        depth: int,
        slot_tiebreak: dict[SlotPosition, float] | None = None,
        first_pos: SlotPosition | None = None,
    ) -> "Board":
        """Recursive helper for `solve()` (TODO)."""
        self.solve_stats.boards_checked += 1

        # Update max depth reached
        self.solve_stats.max_depth_reached = max(self.solve_stats.max_depth_reached, depth)

        # Display progress every 10,000 boards checked
        if self.solve_stats.boards_checked % REPORT_INTERVAL == 0:
            elapsed_time = time() - self.solve_stats.start_time
            print(
                f"Checked {int_comma(self.solve_stats.boards_checked)} board states "
                f"after {time_str(elapsed_time)}; max depth {self.solve_stats.max_depth_reached}, "
                f"current depth {depth}; max filled slots {self.solve_stats.max_filled_slots}.",
                flush=True,
            )

        def _slot_sort_key(pos: SlotPosition) -> tuple[int, int, int, float]:
            """Key function for sorting slots to fill.

            Args:
                pos: Position of the slot to generate the sort key for.

            Returns:
                A tuple representing the sort key for the slot at `pos`.
            """
            slot = self.slot_map.get(pos)
            if slot is None:
                raise RuntimeError(f"No slot found at position {pos} during sorting.")

            rand = 0.0 if slot_tiebreak is None else slot_tiebreak[pos]
            return (
                (
                    -slot_degrees[pos],
                    slot.domain_size,
                    0, #manhattan_distances[pos],
                    rand,  # Is 0.0 if randomization is disabled
                )
                if depth < 4
                else (
                    slot.domain_size,
                    -slot_degrees[pos],
                    0,  # manhattan_distances[pos],
                    rand,  # Is 0.0 if randomization is disabled
                )
            )

        def _candidate_word_order(pos: SlotPosition) -> array[int]:
            """Order candidate word indexes for `pos`.

            Prioritize words which place dictionary-rarer letters earlier.
            Non-intersection placements are weighted higher than intersections.
            This is a value-ordering heuristic only (does not prune domains).
            """
            slot = self.slot_map[pos]

            if slot.domain_size <= 1:
                return array("I", slot.domain.search(1))

            # Only score letters that would actually be placed (i.e., open board cells).
            # Weight non-intersection placements higher than intersections.
            cell_idxs = self._slot_cell_idxs[pos]
            open_positions: list[tuple[int, int]] = []
            dot = ord(".")
            for i in range(slot.length):
                if self.layout[cell_idxs[i]] != dot:
                    continue
                weight = (
                    2
                    if slot.intersections[i] is None #or slot.intersections[i].other_slot.fixed
                    else 1
                )
                open_positions.append((i, weight))
            if not open_positions:
                return array("I", slot.domain.search(1))

            ord_a = ord("A")

            # Secondary value ordering heuristic: prefer words that keep intersecting
            # slots flexible under current domains (sum of resulting domain sizes).
            #
            # Efficient implementation: per intersection, precompute a 26-entry table
            # of |D_y ∩ bucket(letter)| once, then score each word via O(#inters) lookups.
            active_intersections: list[tuple[int, list[int]]] = []
            for i in range(slot.length):
                if self.layout[cell_idxs[i]] != dot:
                    continue  # Not placing a tile here; this word doesn't constrain neighbors.
                inter = slot.intersections[i]
                if inter is None:
                    continue
                other_slot = inter.other_slot
                j = inter.index_in_other_slot
                table = [0] * 26
                buckets = WORD_BUCKETS[other_slot.length][j]
                other_domain = other_slot.domain
                for ch_index in range(26):
                    table[ch_index] = count_and(other_domain, buckets[ch_index])
                active_intersections.append((i, table))

            scored: list[tuple[float, int, int]] = []
            for word_idx in slot.domain.search(1):
                w = WORDS_BY_LENGTH[slot.length][word_idx]
                score = 1.0
                for i, weight in open_positions:
                    idx = ord(w[i]) - ord_a
                    score += weight * tile_bag[idx] * LETTER_RARITY_WEIGHT[idx]

                # Keep-flexibility heuristic: larger is better.
                flexibility = 0
                for i, table in active_intersections:
                    flexibility += table[ord(w[i]) - ord_a]

                scored.append((score, flexibility, word_idx))

            key = lambda t: (-t[1], -t[0], t[2]) if depth < 3 else (-t[0], -t[1], t[2])

            scored.sort(key=key)
            return array("I", (word_idx for _, _, word_idx in scored))

        def _tile_playable_domain_size(pos: SlotPosition, cutoff: int | None = None) -> int:
            """Count how many domain words are playable under the current tile bag.

            This is used as a heuristic only (does not mutate domains).
            The count may be cut off early if it exceeds `cutoff`, which is useful for
            finding the minimum over multiple slots.

            Args:
                pos: Slot position to evaluate.
                cutoff: Optional early-exit threshold. If the count exceeds cutoff, return early.
            """
            slot = self.slot_map[pos]
            if slot.pos is None:
                raise RuntimeError("Uninitialized position in slot.")

            cell_idxs = self._slot_cell_idxs[pos]
            open_positions = [i for i, idx in enumerate(cell_idxs) if self.layout[idx] == ord(".")]

            # If this slot has no open cells, it should not be considered unfilled.
            if not open_positions:
                return slot.domain_size

            playable = 0
            for word_idx in slot.domain.search(1):
                w = WORDS_BY_LENGTH[slot.length][word_idx]
                counts: dict[int, int] = {}
                ok = True
                for i in open_positions:
                    tile_idx = ord(w[i]) - ord("A")
                    new_count = counts.get(tile_idx, 0) + 1
                    if new_count > tile_bag[tile_idx]:
                        ok = False
                        break
                    counts[tile_idx] = new_count
                if ok:
                    playable += 1
                    if cutoff is not None and playable > cutoff:
                        return playable
            return playable

        def _tile_aware_candidate_count(unfilled_count: int) -> int:
            """Choose how many MRV-ranked slots to consider for tile-aware selection.

            We keep this small early (tiles relatively free) and expand later when tiles become
            tight, using filled-slot progress as a proxy.
            """
            if unfilled_count <= 0:
                return 0
            total_slots = len(self.slot_map)
            filled_slots = total_slots - unfilled_count
            progress = filled_slots / total_slots if total_slots else 0.0

            if unfilled_count <= TILE_AWARE_MRV_CANDIDATES:
                return unfilled_count

            if progress < 0.2:
                k = 1
            elif progress < 0.4:
                k = TILE_AWARE_MRV_CANDIDATES
            elif progress < 0.60:
                k = TILE_AWARE_MRV_CANDIDATES * 2
            else:
                # Late in search: slots are fewer and tiles are usually tight.
                k = unfilled_count  # Consider all

            return min(unfilled_count, k)

        def _revise(x_pos: SlotPosition, y_pos: SlotPosition) -> bool:
            """Revise x's domain to be arc-consistent with y for the directed arc x -> y.

            Args:
                x_pos: Position of slot x.
                y_pos: Position of slot y.

            Returns:
                True if x's domain was changed, False otherwise.
            """
            # Find intersection indices (index i in slot_x is index j in slot_y)
            ij = self.constraint_graph.arc_map.get((x_pos, y_pos))
            if ij is None:
                return False
            i, j = ij
            slot_x = self.slot_map[x_pos]
            slot_y = self.slot_map[y_pos]

            # Which letters can slot y apply at position j?
            allowed = [False] * 26
            for ch_index in range(26):
                if (slot_y.domain & WORD_BUCKETS[slot_y.length][j][ch_index]).any():
                    allowed[ch_index] = True

            if all(allowed):
                return False  # Domain unchanged

            # Reduce slot x's domain by removing words whose letter at position i
            # is not allowed by slot y (at position j).
            forbidden = zeros(len(slot_x.domain))
            for ch_index, is_allowed in enumerate(allowed):
                if not is_allowed:
                    forbidden |= WORD_BUCKETS[slot_x.length][i][ch_index]

            # Compare new domain to old domain
            new_domain = slot_x.domain & ~forbidden
            if new_domain == slot_x.domain:
                return False  # Domain unchanged

            # Update domain and record undo information
            self._undo_trails.domain_changes.append((x_pos, slot_x.domain, slot_x.domain_size))
            slot_x.domain = new_domain
            slot_x.domain_size = new_domain.count()
            return True

        def _ac3(seed_pos: SlotPosition) -> bool:
            """Run AC-3 propagation seeded by a recent assignment at seed_pos.

            Args:
                seed_pos: Position of the slot that was most recently assigned a word.

            Returns:
                True if arc-consistency was achieved without empty domains, False otherwise.
            """
            q: deque[tuple[SlotPosition, SlotPosition]] = deque()
            for neighbor_pos, _, _ in self.constraint_graph.neighbors.get(seed_pos, []):
                q.append((neighbor_pos, seed_pos))

            while q:
                x_pos, y_pos = q.popleft()
                if _revise(x_pos, y_pos):
                    if self.slot_map[x_pos].domain_size == 0:
                        return False
                    for z_pos, _, _ in self.constraint_graph.neighbors.get(x_pos, []):
                        if z_pos != y_pos:
                            q.append((z_pos, x_pos))
            return True

        # Identify which slots still have '.' cells on the board.
        if not self._unfilled_slots:
            return self

        # Filled slots is derivable from unfilled slots, and naturally includes forced plays.
        n_filled = len(self.slot_map) - len(self._unfilled_slots)
        if n_filled > self.solve_stats.max_filled_slots:
            self.solve_stats.max_filled_slots = n_filled
            if self.solve_stats.max_filled_slots > REPORT_HI_SCORE_MIN:
                print(f"New high: {n_filled} slots filled, current layout:")
                self.print()
                print(f"Remaining tiles: {tile_bag_to_string(tile_bag)}")

        # Marker to undo any placements performed within this recursion frame (including
        # forced singleton plays) before backtracking to the caller.
        frame_mark = _UndoInfo(
            layout_mark=len(self._undo_trails.layout_changes),
            tile_mark=len(self._undo_trails.tile_decrements),
            domain_mark=len(self._undo_trails.domain_changes),
            filled_mark=len(self._undo_trails.filled_changes),
        )

        # Fail fast if any unfilled slot has no candidates.
        for pos in self._unfilled_slots:
            if self.slot_map[pos].domain_size <= 0:
                raise ValueError("Encountered an empty domain during solve.")

        # Play all slots with a single candidate word, fail-fast if any domain becomes empty.
        made_progress = True
        while made_progress:
            made_progress = False
            for pos in list(self._unfilled_slots):
                slot = self.slot_map[pos]
                if slot.domain_size == 1:
                    word_idx = slot.domain.find(1)
                    word = WORDS_BY_LENGTH[slot.length][word_idx]
                    try:
                        self.place_word(pos, word, tile_bag)
                        if not _ac3(pos):
                            raise ValueError("Encountered an empty domain during solve.")
                    except ValueError:
                        # Any failure means this recursion frame is inconsistent; revert all
                        # forced moves made in this frame and backtrack.
                        self.undo_place_word(frame_mark, tile_bag)
                        raise

                    made_progress = True

        # Forced plays may have increased filled slots without increasing recursion depth.
        n_filled = len(self.slot_map) - len(self._unfilled_slots)
        if n_filled > self.solve_stats.max_filled_slots:
            self.solve_stats.max_filled_slots = n_filled
            if self.solve_stats.max_filled_slots > REPORT_HI_SCORE_MIN:
                print(f"New high: {n_filled} slots filled, current layout:")
                self.print()
                print(f"Remaining tiles: {tile_bag_to_string(tile_bag)}")

        # Select the next slot to fill
        if depth == 0 and first_pos is not None and first_pos in self._unfilled_slots:
            pos = first_pos
        else:
            ranked = sorted(self._unfilled_slots, key=_slot_sort_key)
            k = _tile_aware_candidate_count(len(self._unfilled_slots))
            candidates = ranked[:k]

            # Search top candidates for the one with fewest playable words under current tile bag
            pos = candidates[0]
            best_playable = _tile_playable_domain_size(pos)
            for other in candidates[1:]:
                playable = _tile_playable_domain_size(other, cutoff=best_playable)
                if playable < best_playable:
                    pos = other
                    best_playable = playable
                elif playable == best_playable and _slot_sort_key(other) < _slot_sort_key(pos):
                    pos = other

            if best_playable <= 0:
                self.undo_place_word(frame_mark, tile_bag)
                raise ValueError(f"No playable words remain for slot at {pos} under current tiles.")

        slot = self.slot_map[pos]

        # Try each word in the slot's domain
        for word_idx in _candidate_word_order(pos):
            word = WORDS_BY_LENGTH[slot.length][word_idx]
            try:
                undo_info = self.place_word(pos, word, tile_bag)
                if depth == 0:
                    print(f"Playing initial word '{word}' at {pos}...")
            except ValueError:
                # Placement failed; try next word
                continue

            try:
                if not _ac3(pos):
                    # AC-3 failed; undo and try next word
                    self.undo_place_word(undo_info, tile_bag)
                    continue
                return self.solve_helper(tile_bag, manhattan_distances, slot_degrees, depth + 1)
            except ValueError:
                # Recursive call failed; undo and try next word
                self.undo_place_word(undo_info, tile_bag)
                continue

        # Failed at current depth; undo forced moves and backtrack.
        self.undo_place_word(frame_mark, tile_bag)
        raise ValueError(f"No solution found when trying to fill slot at {pos}.")


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

    # ACROSS LOOK intersects DOWN BOO at LOOK[1] == BOO[1]
    assert board.constraint_graph.arc_map[
        (SlotPosition(Direction.ACROSS, 1, 0), SlotPosition(Direction.DOWN, 0, 1))
    ] == (1, 1)
    # DOWN .O. intersects ACROSS LOOK at .O.[1] == LOOK[2]
    assert board.constraint_graph.arc_map[
        (SlotPosition(Direction.DOWN, 0, 2), SlotPosition(Direction.ACROSS, 1, 0))
    ] == (1, 2)

    # Check neighbor lists for LOOK and BOO, which intersect at index 1 of both slots
    assert (SlotPosition(Direction.DOWN, 0, 1), 1, 1) in board.constraint_graph.neighbors[
        SlotPosition(Direction.ACROSS, 1, 0)
    ]
    assert (SlotPosition(Direction.ACROSS, 1, 0), 1, 1) in board.constraint_graph.neighbors[
        SlotPosition(Direction.DOWN, 0, 1)
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


def test_solver(grid_rows: int, grid_cols: int, tile_copies: int):
    """Test the solver on an empty board.

    Args:
        grid_rows: Number of rows in the board.
        grid_cols: Number of columns in the board.
        tile_copies: Number of copies of each letter tile A-Z in the tile bag.
    """
    layout = "".join(
        ["." * grid_cols for _ in range(grid_rows)]
    )  # empty board of size GRID_ROWS x GRID_COLS
    board = Board(layout, grid_rows, grid_cols)

    tile_bag = create_tile_bag("ABCDEFGHIJKLMNOPQRSTUVWXYZ" * tile_copies)

    try:
        solved_board = board.solve(tile_bag, SlotPosition(Direction.ACROSS, 0, 0))
        print("Solved board:")
        solved_board.print()
        print(f"Boards checked: {int_comma(solved_board.solve_stats.boards_checked)}")
        print(f"Time taken: {time_str(time() - solved_board.solve_stats.start_time)}")
    except ValueError:
        print("No solution found.")


if __name__ == "__main__":
    n_rows, n_cols, n_copies = map(int, sys.argv[1:4])
    test_solver(n_rows, n_cols, n_copies)
