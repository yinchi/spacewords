"""Board representation for the Spacewords game."""

from collections import defaultdict, deque
from typing import NamedTuple

from bitarray import bitarray
from bitarray.util import zeros

from spacewords2.slot import Direction, Intersection, Slot, SlotPosition
from spacewords2.tiles import TileBag, create_tile_bag
from spacewords2.words import WORD_BUCKETS, WORD_INDEXES, WORDS_BY_LENGTH


class _UndoInfo(NamedTuple):
    """Marker into the undo trails for a single move.

    Undo will reverse changes until the remaining lengths of the trails match these marks.
    """

    layout_mark: int
    tile_mark: int
    domain_mark: int


class _UndoTrails(NamedTuple):
    """Trails for undoing changes to the board and tile bag."""

    layout_changes: list[tuple[int, int]]
    """List of (index, old_byte) tuples representing changes to the board layout."""

    tile_decrements: list[int]
    """List of tile indices that were decremented in the tile bag."""

    domain_changes: list[tuple[SlotPosition, bitarray, int]]
    """List of (slot_position, old_domain, old_domain_size) tuples.

    Each tuple represents changes to slot domains.
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
        self._undo_trails: _UndoTrails = _UndoTrails([], [], [])

        self.intersection_map: defaultdict[tuple[int, int], list[Intersection]] = defaultdict(list)
        """Mapping from cell positions (row, col) to a list of
        (slot_start, index_in_slot) entries."""

        self.boards_checked: int = 0
        """Number of board states checked during solving."""

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

        self.constraint_graph = self._build_constraint_graph()
        """Precomputed constraint graph for the board.

        Since board topology does not change, this is built once at initialization.
        """

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
        )

        changed_domains: set[SlotPosition] = set()

        try:
            # Cheap pre-check: validate conflicts and ensure the tile bag can cover all new
            # letters needed for this placement, without mutating anything.
            direction, start_row, start_col = slot.pos
            base = start_row * self.n_cols + start_col
            pending_writes: list[tuple[int, int]] = []
            needed_tiles = [0] * 26

            for i in range(slot.length):
                if direction == Direction.ACROSS:
                    idx = base + i
                else:
                    idx = base + i * self.n_cols

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

    def solve(self, tile_bag: TileBag, first_pos: SlotPosition) -> "Board":
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

        Raises:
            ValueError: If the board cannot be solved with the given tiles.
        """

        def _manhattan_distance(pos: SlotPosition) -> int:
            """Get the Manhattan distance from `first_pos` to `pos`."""
            r1, c1 = pos.start_row, pos.start_col
            r2, c2 = first_pos.start_row, first_pos.start_col
            return abs(r1 - r2) + abs(c1 - c2)

        slot_degrees: dict[SlotPosition, int] = {
            slot_start: sum(1 for inter in slot.intersections if inter is not None)
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

                direction, start_row, start_col = slot.pos
                base = start_row * self.n_cols + start_col

                # Effective availability per letter for this slot: tile bag plus letters
                # already present on the board in this slot.
                effective = [int(x) for x in tile_bag]
                for i in range(slot.length):
                    if direction == Direction.ACROSS:
                        idx = base + i
                    else:
                        idx = base + i * self.n_cols
                    ch_byte = self.layout[idx]
                    if ord("A") <= ch_byte <= ord("Z"):
                        effective[ch_byte - ord("A")] += 1

                to_clear: list[int] = []  # Indexes of unplacable words
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

        self.solve_helper(tile_bag, manhattan_distances, slot_degrees)
        return self

    def solve_helper(
        self,
        tile_bag: TileBag,
        manhattan_distances: dict[SlotPosition, int],
        slot_degrees: dict[SlotPosition, int],
    ) -> "Board":
        """Recursive helper for `solve()` (TODO)."""
        if self.boards_checked % 10000 == 0:
            print(f"Checked {self.boards_checked} board states...")

        def _slot_sort_key(pos: SlotPosition) -> tuple[int, int, int, int, int]:
            """Key function for sorting slots to fill.

            When backtracking, slots to attempt to fill are chosen in order of:

            1. Manhattan distance from `first_pos`.  This "flood-fills" out from the first slot and
               increases the chance of domain reductions from intersections.  Ensures the slot at
               `first_pos` is attempted first.  Since distances are pre-computed, we don't need to
               pass `first_pos` here.
            2. Fewest remaining possible words in the slot's domain (MRV heuristic).
            3. Largest number of intersections with other slots (degree heuristic).
            4. Across slots before down slots.
            5. Top to bottom, left to right.

            Args:
                pos: Position of the slot to generate the sort key for.

            Returns:
                A tuple representing the sort key for the slot at `pos`.
            """
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

        def _slot_filled_on_board(pos: SlotPosition) -> bool:
            """Check if the slot at `pos` is already filled on the board (no '.' cells).

            If the slot is filled, its domain should already be fixed to a single word, but
            this is not checked here (only checks the board layout).

            Args:
                pos: Position of the slot to check.

            Returns:
                True if the slot is completely filled on the board, False otherwise.
            """
            slot = self.slot_map[pos]
            if slot.pos is None:
                raise RuntimeError("Uninitialized position in slot.")
            direction, start_row, start_col = slot.pos
            base = start_row * self.n_cols + start_col
            for i in range(slot.length):
                if direction == Direction.ACROSS:
                    idx = base + i
                else:
                    idx = base + i * self.n_cols
                if self.layout[idx] == ord("."):
                    return False
            return True

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
        unfilled = [pos for pos in self.slot_map if not _slot_filled_on_board(pos)]
        if not unfilled:
            return self

        # Fail fast if any unfilled slot has no candidates.
        for pos in unfilled:
            if self.slot_map[pos].domain_size <= 0:
                raise ValueError("Encountered an empty domain during solve.")

        self.boards_checked += 1

        # Select the next slot to fill
        pos = min(unfilled, key=_slot_sort_key)
        slot = self.slot_map[pos]

        # Try each word in the slot's domain
        for word_idx in slot.domain.search(1):
            word = WORDS_BY_LENGTH[slot.length][word_idx]
            try:
                undo_info = self.place_word(pos, word, tile_bag)
            except ValueError:
                continue

            try:
                if not _ac3(pos):
                    self.undo_place_word(undo_info, tile_bag)
                    continue
                return self.solve_helper(tile_bag, manhattan_distances, slot_degrees)
            except ValueError:
                self.undo_place_word(undo_info, tile_bag)
                continue

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


def test_solver():
    """Test the solver on a simple board."""
    grid_rows = 5
    grid_cols = 6
    layout = "".join(
        ["." * grid_cols for _ in range(grid_rows)]
    )  # empty board of size GRID_ROWS x GRID_COLS
    board = Board(layout, grid_rows, grid_cols)

    tile_bag = create_tile_bag("ABCDEFGHIJKLMNOPQRSTUVWXYZ" * grid_rows)

    try:
        solved_board = board.solve(tile_bag, SlotPosition(Direction.ACROSS, 0, 0))
        print("Solved board:")
        solved_board.print()
        print(f"Boards checked: {solved_board.boards_checked}")
    except ValueError:
        print("No solution found.")


if __name__ == "__main__":
    test_solver()
