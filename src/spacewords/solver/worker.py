"""Main module for worker tasks in the parallel solver."""

from collections import Counter, deque
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from math import prod
from multiprocessing.sharedctypes import Synchronized
from time import time
from typing import TypeAlias

from spacewords.board import Board
from spacewords.puzzle_config import PuzzleConfig
from spacewords.solver.config import config as solver_config
from spacewords.solver.utils import (
    Direction,
    get_slot_indices,
    get_slot_words,
    get_word_counter,
    is_playable,
    validate_board,
)

Slot: TypeAlias = tuple[Direction, int, int]
Domain: TypeAlias = Collection[str]
SlotWords: TypeAlias = Mapping[Slot, Domain]
PlacementUndo: TypeAlias = tuple[list[tuple[int, str]], list[str]]


@dataclass(kw_only=True)
class WorkerState:
    """Global state maintained by each worker process."""

    worker_idx: int
    """Index of the worker process."""

    total_fillable_cells: int
    """Total number of fillable cells in the puzzle."""

    start_time: float
    """Timestamp when the worker started, in seconds since the epoch."""

    initial_word_map: dict[tuple[int, int, str], set[str]]
    """Initial word map loaded by the worker."""

    initial_letter_frequency: dict[str, float]
    """Initial letter frequency loaded by the worker."""

    words: set[str]
    """Set of valid words for this puzzle (post tile-filtering)."""

    words_by_length: dict[int, set[str]]
    """Words bucketed by length for fast slot-domain construction."""

    slot_indices: dict[Slot, list[int]] | None = None
    """Cached slot->cell-indices mapping (depends only on board blocks)."""

    across_slot_for_cell: dict[int, Slot] | None = None
    """Cached mapping from cell index to ACROSS slot."""

    down_slot_for_cell: dict[int, Slot] | None = None
    """Cached mapping from cell index to DOWN slot."""

    n_boards_examined: int = 0
    """Number of boards examined by this worker."""

    last_threshold_reported: float = -1
    """Last threshold (number of filled cells) reported by this worker."""


worker_state: WorkerState | None = None
"""Global state for each worker process."""


def init_worker_globals(
    worker_ctr: Synchronized[int],
    total_fillable_cells: int,
    start_time: float,
    initial_word_map: dict[tuple[int, int, str], set[str]],
    initial_letter_frequency: dict[str, float],
    words: set[str],
) -> None:
    """Initialize global variables for worker processes.

    Args:
        worker_ctr (Synchronized[int]): Shared counter for workers.
        total_fillable_cells (int): Total number of fillable cells in the puzzle.
        start_time (float): UNIX timestamp when the solver started.
        initial_word_map (dict[tuple[int, int, str], set[str]]): Initial word map loaded by
            the worker.
        initial_letter_frequency (dict[str, float]): Initial letter frequency loaded by the worker.
        words (set[str]): Set of valid words filtered for this puzzle.
    """
    global worker_state  # noqa: PLW0603
    with worker_ctr.get_lock():
        # Get and set the shared worker counter atomically, using the obtained value
        # as the worker index
        worker_idx = worker_ctr.value
        worker_ctr.value += 1

    words_by_length: dict[int, set[str]] = {}
    for w in words:
        words_by_length.setdefault(len(w), set()).add(w)
    worker_state = WorkerState(
        worker_idx=worker_idx,
        total_fillable_cells=total_fillable_cells,
        start_time=start_time,
        initial_word_map=initial_word_map,
        initial_letter_frequency=initial_letter_frequency,
        words=words,
        words_by_length=words_by_length,
    )
    print(f"Worker {worker_state.worker_idx} initialized.")


def worker_task(anchor_word: str, puzzle_config: dict) -> Board | None:
    """Worker task to solve the puzzle with a given anchor word.

    Args:
        anchor_word (str): The anchor word to place.
        puzzle_config (dict): Dict representation of a PuzzleConfig.

    Returns:
        A solved Board if a solution is found, else None.
    """
    # Ensure worker_state is initialized
    if not worker_state:
        raise RuntimeError("Worker state not initialized. Call init_worker_globals first.")

    print(
        f"Worker {worker_state.worker_idx} starting with anchor word: {anchor_word}",
        flush=True,
    )

    puzzle_config = PuzzleConfig.from_dict(puzzle_config)
    board = Board(puzzle_config.board_str, *puzzle_config.dims)
    words = worker_state.words
    words_by_length = worker_state.words_by_length
    word_map = worker_state.initial_word_map
    letter_frequency = worker_state.initial_letter_frequency

    available_tiles = puzzle_config.tiles.copy()

    # Place the anchor word on the board.
    anchor_col = solver_config.anchor_col
    for row in range(board.n_rows):
        if board[row, anchor_col] == ".":
            board[row, anchor_col] = anchor_word[row]
        if board[row, anchor_col] != anchor_word[row]:
            print(
                f"Worker {worker_state.worker_idx}, {anchor_word}: "
                f"conflict placing anchor word at row {row}, col {anchor_col}.",
                flush=True,
            )
            return None  # Conflict placing anchor word

    # Validate that the board so far is playable, i.e. tiles needed is a subset of available tiles.
    tiles_needed = Counter(board.data)
    del tiles_needed["#"]  # Remove blocked cells from counter
    del tiles_needed["."]  # Remove empty cells from counter
    valid_board = tiles_needed <= available_tiles
    if not valid_board:
        return None  # Not playable

    # Compute unused tiles after placing the anchor word (and any other words already placed).
    unused_tiles = available_tiles - tiles_needed

    # Build (or reuse) slot indices mapping and cell->slot lookups.
    if worker_state.slot_indices is None:
        slot_indices = get_slot_indices(board)
        across_slot_for_cell: dict[int, Slot] = {}
        down_slot_for_cell: dict[int, Slot] = {}
        for slot, indices in slot_indices.items():
            direction, _start_idx, _length = slot
            if direction == Direction.ACROSS:
                for idx in indices:
                    across_slot_for_cell[idx] = slot
            else:
                for idx in indices:
                    down_slot_for_cell[idx] = slot
        worker_state.slot_indices = slot_indices
        worker_state.across_slot_for_cell = across_slot_for_cell
        worker_state.down_slot_for_cell = down_slot_for_cell
    else:
        slot_indices = worker_state.slot_indices
        across_slot_for_cell = worker_state.across_slot_for_cell or {}
        down_slot_for_cell = worker_state.down_slot_for_cell or {}
    print(
        f"Worker {worker_state.worker_idx}, {anchor_word}: computed slot indices.",
        flush=True,
    )

    # Compute possible words for each slot (direction, index, length).
    slot_words = get_slot_words(
        board,
        slot_indices,
        words,
        word_map,
        unused_tiles,
        words_by_length=words_by_length,
    )
    # print(
    #     f"Worker {worker_state.worker_idx}, {anchor_word}: "
    #     f"{[{slot: len(words)} for slot, words in slot_words.items()]}.",
    #     flush=True,
    # )

    if any(len(word_set) == 0 for word_set in slot_words.values()):
        print(
            f"Worker {worker_state.worker_idx}, {anchor_word}: no possible words for some slots.",
            flush=True,
        )
        return None  # No possible words for some slots

    # Print statistics about slot word domains
    # A domain is the set of possible words for a slot
    sum_domain_sizes = sum(len(word_set) for word_set in slot_words.values())
    min_domain_size = min(len(word_set) for word_set in slot_words.values())
    max_domain_size = max(len(word_set) for word_set in slot_words.values())
    print(
        f"Worker {worker_state.worker_idx}, {anchor_word}: "
        f"Total size of domains = {sum_domain_sizes}, "
        f"Min. domain size = {min_domain_size}, "
        f"Max. domain size = {max_domain_size}.",
        flush=True,
    )

    # Call Constraint Programming solver
    solution = solve_with_cp(
        board=board,
        slot_words=slot_words,
        all_words=words,
        word_map=word_map,
        unused_tiles=unused_tiles,
        letter_frequency=letter_frequency,
        worker_state=worker_state,
        slot_idxs=slot_indices,
        across_slot_for_cell=across_slot_for_cell,
        down_slot_for_cell=down_slot_for_cell,
    )

    if solution is not None:
        print(
            f"Worker {worker_state.worker_idx}, {anchor_word}: solution found!",
            flush=True,
        )
    else:
        print(
            f"Worker {worker_state.worker_idx}, {anchor_word}: no solution found.",
            flush=True,
        )

    return solution


class InvalidPlacementError(Exception):
    """Exception raised for invalid word placements on the board."""

    pass


def solve_with_cp(
    board: Board,
    *,
    slot_words: SlotWords,
    all_words: set[str],
    word_map: dict[tuple[int, int, str], set[str]],
    unused_tiles: Counter[str],
    letter_frequency: dict[str, float],
    worker_state: WorkerState,
    slot_idxs: dict[Slot, list[int]],
    across_slot_for_cell: dict[int, Slot],
    down_slot_for_cell: dict[int, Slot],
    depth: int = 0,
) -> Board | None:
    """Solve the puzzle using Constraint Programming.

    Args:
        board (Board): The current state of the board.
        slot_words (SlotWords): Mapping of slots to a domain of possible words.
            Updated as words are placed on the board.
        all_words (set[str]): Set of all valid words.
            Fixed for the duration of the solving process.
        word_map (dict[tuple[int, int, str], set[str]]): Mapping of `(word_length, position_in_word,
            character)` to sets of words. Used for fast fixed-letter pruning during constraint
            propagation.
        unused_tiles (Counter[str]): Counter of unused tiles.
            Updated as words are placed on the board.
        letter_frequency (dict[str, float]): Frequency of each letter in the word list.
            Used to prioritize word choices when applying depth-first search.
        worker_state (WorkerState): The state of the worker.
        slot_idxs (dict[Slot, list[int]]): Mapping of slots to their cell indices.
        across_slot_for_cell (dict[int, Slot]): Mapping of cell index to the ACROSS slot.
        down_slot_for_cell (dict[int, Slot]): Mapping of cell index to the DOWN slot.
        depth (int): Recursion depth (0 = top-level call).

    Returns:
        A solved Board if a solution is found, else None.
    """
    if depth == 0:
        worker_state.last_threshold_reported = -1  # Reset at top level
    worker_state.n_boards_examined += 1

    # Apply constraint propagation to reduce domains
    # (slot_words is a mapping of each slot to a domain of possible words)
    updated_slot_words = apply_constraint_propagation(
        board,
        slot_words=slot_words,
        available_tiles=unused_tiles,
        slot_idxs=slot_idxs,
        word_map=word_map,
    )

    # Check for termination conditions
    # A solved board has no remaining slots to fill, thus an empty dict (no elements)
    if not updated_slot_words:
        # Validate the board to ensure all words are valid
        if validate_board(board, word_list=all_words):
            return board  # Solved
    # An impossible board has some slot with no possible words (element is None or with length 0)
    if any(not words for words in updated_slot_words.values()):
        return None

    # Otherwise, continue solving recursively
    # Otherwise, continue solving recursively.
    # IMPORTANT: this function mutates `board` and `unused_tiles` in-place, and must undo
    # all mutations before returning None.

    forced_undos: list[PlacementUndo] = []

    # First, apply forced placements
    n_forced_moves = 0

    def _first(x: Iterable[str]) -> str:
        return next(iter(x))

    while True:
        forced_moves = [
            (slot, _first(words)) for slot, words in updated_slot_words.items() if len(words) == 1
        ]
        if not forced_moves:
            break

        # Place forced moves, checking at each step for validity
        # (a forced move may become invalid due to previous forced moves)
        for slot, word in forced_moves:
            n_forced_moves += 1

            try:
                forced_undos.append(
                    place_word_inplace(
                        board,
                        slot,
                        word,
                        word_list=all_words,
                        available_tiles=unused_tiles,
                        slot_idxs=slot_idxs,
                        across_slot_for_cell=across_slot_for_cell,
                        down_slot_for_cell=down_slot_for_cell,
                    )
                )
            except InvalidPlacementError:
                undo_placements(board, unused_tiles, forced_undos)
                return None  # Forced placement invalid, backtrack

            # Remove the slot from consideration
            del updated_slot_words[slot]

        # Update slot words after forced placements
        # This may lead to new forced placements, so we do this inside the while loop
        updated_slot_words = apply_constraint_propagation(
            board=board,
            slot_words=updated_slot_words,
            available_tiles=unused_tiles,
            slot_idxs=slot_idxs,
            word_map=word_map,
        )

    # Report progress if reporting threshold met (defined in imported solver_config)
    report_progress(
        worker_state,
        board,
        updated_slot_words,
        available_tiles=unused_tiles,
    )

    # Check for termination conditions
    # A solved board has no remaining slots to fill, thus an empty dict (no elements)
    if not updated_slot_words:
        # Validate the board to ensure all words are valid
        if validate_board(board, word_list=all_words):
            return board  # Solved
    # An impossible board has some slot with no possible words (element is None or with length 0)
    if any(not words for words in updated_slot_words.values()):
        undo_placements(board, unused_tiles, forced_undos)
        return None

    # Select the next slot to fill using a priority function
    min_domain_size = min(len(words) for words in updated_slot_words.values())
    min_domain_items: list[tuple[Slot, Domain]] = [
        (slot, words)
        for slot, words in updated_slot_words.items()
        if len(words) == min_domain_size
    ]

    def word_rarity_score(word: str, tiles: Counter[str]) -> float:
        # Higher score => consumes rarer (lower-availability) tiles.
        # This is a heuristic tie-breaker; domains are already filtered to be playable.
        score = 0.0
        for ch, cnt in get_word_counter(word).items():
            avail = tiles.get(ch, 0)
            if avail <= 0:
                # Shouldn't happen (unplayable), but keep ordering deterministic.
                return float("inf")
            score += cnt / float(avail)
        return score

    def slot_tiebreak_key(slot_word_item: tuple[Slot, Domain]) -> tuple[float, int, int, Slot]:
        slot, words = slot_word_item
        direction, start_idx, _ = slot
        row, col = board.get_2d_idx(start_idx)
        distance = row + abs(col - solver_config.anchor_col)  # Manhattan distance
        direction_pref = 0 if direction == Direction.ACROSS else 1

        rarity_potential = 0.0
        if solver_config.prefer_rare_letters_in_slot_order and len(min_domain_items) > 1:
            # Only evaluate rarity for MRV ties to keep overhead low.
            rarity_potential = max((word_rarity_score(w, unused_tiles) for w in words), default=0.0)

        # We choose min() over this key, so negate rarity to prefer higher rarity.
        return (-rarity_potential, distance, direction_pref, slot)

    slot, possible_words = min(min_domain_items, key=slot_tiebreak_key)
    possible_words = list(possible_words)

    # Score possible words by letter frequency (lowest score first) and sort.
    # For deterministic runs, use the word itself as a stable tiebreaker.
    # Score is the product of frequencies of letters in the word.
    if solver_config.deterministic:
        possible_words.sort(key=lambda w: (prod(letter_frequency[ch] for ch in w), w))
    else:
        possible_words.sort(key=lambda w: prod(letter_frequency[ch] for ch in w))

    # Try each possible word for the selected slot
    for word in possible_words:
        try:
            placement_undo = place_word_inplace(
                board,
                slot,
                word,
                word_list=all_words,
                available_tiles=unused_tiles,
                slot_idxs=slot_idxs,
                across_slot_for_cell=across_slot_for_cell,
                down_slot_for_cell=down_slot_for_cell,
            )
        except InvalidPlacementError:
            continue  # Invalid placement, try next word

        # Recursively solve with the updated board and tile set.
        # Avoid allocating a new dict each branch; remove the chosen slot in-place and restore.
        removed_domain = updated_slot_words.pop(slot)
        try:
            solution = solve_with_cp(
                board,
                slot_words=updated_slot_words,
                all_words=all_words,
                word_map=word_map,
                unused_tiles=unused_tiles,
                letter_frequency=letter_frequency,
                worker_state=worker_state,
                slot_idxs=slot_idxs,
                across_slot_for_cell=across_slot_for_cell,
                down_slot_for_cell=down_slot_for_cell,
                depth=depth + 1,
            )
        finally:
            updated_slot_words[slot] = removed_domain
        if solution is not None:
            return solution  # Solution found

        undo_placements(board, unused_tiles, [placement_undo])

    # No solution found for any word in this slot, backtrack
    undo_placements(board, unused_tiles, forced_undos)
    return None


def undo_placements(
    board: Board, available_tiles: Counter[str], undos: Iterable[PlacementUndo]
) -> None:
    """Undo one or more in-place placements."""
    # Undo in reverse order to restore the board to the exact prior state.
    for board_changes, tile_decrements in reversed(list(undos)):
        for ch in tile_decrements:
            available_tiles[ch] += 1
        for cell_idx, old_char in reversed(board_changes):
            board.data[cell_idx] = old_char


def place_word_inplace(
    board: Board,
    slot: Slot,
    word: str,
    *,
    word_list: set[str],
    available_tiles: Counter[str],
    slot_idxs: dict[Slot, list[int]],
    across_slot_for_cell: dict[int, Slot],
    down_slot_for_cell: dict[int, Slot],
) -> PlacementUndo:
    """Place a word on the board by mutating `board`/`available_tiles` in-place.

    Returns an undo record that must be applied if the caller backtracks.
    """
    direction, _start_idx, length = slot
    if len(word) != length:
        raise InvalidPlacementError("Word length does not match slot length.")

    board_changes: list[tuple[int, str]] = []
    tile_decrements: list[str] = []

    try:
        indices = slot_idxs[slot]

        # Place letters into empty cells ('.'), consuming tiles.
        for pos, cell_idx in enumerate(indices):
            word_char = word[pos]
            current_char = board.data[cell_idx]

            if current_char == word_char:
                continue
            if current_char == ".":
                if available_tiles[word_char] <= 0:
                    raise InvalidPlacementError("Insufficient tiles to place the word.")
                board_changes.append((cell_idx, current_char))
                board.data[cell_idx] = word_char
                available_tiles[word_char] -= 1
                tile_decrements.append(word_char)
                continue

            raise InvalidPlacementError("Conflict with existing letter on the board.")

        # Validate crossing words (only when fully filled).
        # Note: we only need to validate crossings for cells in this placed slot.
        for cell_idx in indices:
            if direction == Direction.ACROSS:
                crossing_slot = down_slot_for_cell.get(cell_idx)
            else:
                crossing_slot = across_slot_for_cell.get(cell_idx)

            if crossing_slot is None:
                continue
            _cross_dir, _cross_start_idx, cross_len = crossing_slot
            if cross_len <= 1:
                continue

            cross_word = "".join(board.data[i] for i in slot_idxs[crossing_slot])
            if "." not in cross_word and cross_word not in word_list:
                raise InvalidPlacementError("Crossing word is invalid.")

        return (board_changes, tile_decrements)
    except Exception:
        # Ensure callers never observe partial placements.
        undo_placements(board, available_tiles, [(board_changes, tile_decrements)])
        raise


def apply_constraint_propagation(
    board: Board,
    slot_words: SlotWords,
    available_tiles: Counter[str],
    *,
    slot_idxs: dict[Slot, list[int]],
    word_map: dict[tuple[int, int, str], set[str]],
) -> dict[Slot, set[str]]:
    """Apply constraint propagation to reduce the domains of slot words.

    For each slot, eliminate words from its domain that cannot be placed
    on the board given the current state and available tiles.  Also, for each
    intersection between slots, eliminate words from each slot that would leave no
    possible words for the intersecting slot.

    Uses AC-3 style constraint propagation for intersections.  See:
    https://en.wikipedia.org/wiki/AC-3_algorithm

    Args:
        board (Board): The current state of the board.
        slot_words (SlotWords): Mapping of slots to a domain of possible words.
        available_tiles (Counter[str]): Counter of available tiles.
        slot_idxs (dict[Slot, list[int]]): Mapping of slots to their cell indices.
        word_map (dict[tuple[int, int, str], set[str]]): Mapping of `(word_length, position_in_word,
            character)` to sets of words.

    Returns:
        Updated mapping of slots to possible words after constraint propagation.
        If any slot becomes impossible, returns an all-empty mapping.
    """
    if available_tiles.total() == 0 and "." in board.data:
        return {k: set() for k in slot_words.keys()}  # Early exit if no tiles left
    if any(len(words) == 0 for words in slot_words.values()):
        return {k: set() for k in slot_words.keys()}  # Early exit if any slot has no words

    updated_slot_words: dict[Slot, set[str]] = {
        slot: set(words) for slot, words in slot_words.items()
    }
    active_slots = list(updated_slot_words.keys())
    if solver_config.deterministic:
        active_slots.sort()
    slot_idx_sets: dict[Slot, set[int]] = {slot: set(slot_idxs[slot]) for slot in active_slots}

    # Precompute intersection positions once for the current slot set.
    # This avoids repeated O(n^2) set intersections and index arithmetic inside the main loop.
    pos_by_cell: dict[Slot, dict[int, int]] = {
        slot: {cell_idx: pos for pos, cell_idx in enumerate(slot_idxs[slot])}
        for slot in updated_slot_words
    }

    # Precompute fixed board letters and already-filled letters per slot once.
    # The board does not change during CP, only the slot domains do.
    fixed_positions_by_slot: dict[Slot, list[tuple[int, str]]] = {}
    filled_letters_by_slot: dict[Slot, Counter[str]] = {}
    for slot in active_slots:
        fixed_positions: list[tuple[int, str]] = []
        filled_letters: Counter[str] = Counter()
        for word_pos, board_idx in enumerate(slot_idxs[slot]):
            ch = board.data[board_idx]
            if ch not in (".", "#"):
                fixed_positions.append((word_pos, ch))
                filled_letters[ch] += 1
        fixed_positions_by_slot[slot] = fixed_positions
        filled_letters_by_slot[slot] = filled_letters
    intersection_specs: list[tuple[Slot, Slot, int, int]] = []
    for i in range(len(active_slots)):
        slot_a = active_slots[i]
        for j in range(i + 1, len(active_slots)):
            slot_b = active_slots[j]
            intersections = slot_idx_sets[slot_a] & slot_idx_sets[slot_b]
            if not intersections:
                continue
            pos_a = pos_by_cell[slot_a]
            pos_b = pos_by_cell[slot_b]
            if solver_config.deterministic:
                intersections_iter = sorted(intersections)
            else:
                intersections_iter = intersections
            for cell_idx in intersections_iter:
                intersection_specs.append((slot_a, slot_b, pos_a[cell_idx], pos_b[cell_idx]))

    # Cache of possible letters at each (slot, position). Recomputed only when that slot's
    # domain changes.
    letters_at_pos_cache: dict[Slot, dict[int, set[str]]] = {}
    cache_dirty: dict[Slot, bool] = {slot: True for slot in active_slots}

    def _letters_at_pos(slot: Slot, pos: int) -> set[str]:
        slot_cache = letters_at_pos_cache.get(slot)
        if slot_cache is None:
            slot_cache = {}
            letters_at_pos_cache[slot] = slot_cache

        if cache_dirty.get(slot, True) or pos not in slot_cache:
            slot_cache[pos] = {w[pos] for w in updated_slot_words[slot]}
        cache_dirty[slot] = False
        return slot_cache[pos]

    # Unary pruning: board-fixed letters + tile-feasibility. These depend only on the current
    # board and tile bag, not on other slot domains, so a single pass is sufficient.
    slots_to_check = list(updated_slot_words.keys())
    if solver_config.deterministic:
        slots_to_check.sort()

    for slot in slots_to_check:
        fixed_positions = fixed_positions_by_slot.get(slot, [])
        if fixed_positions:
            n_before = len(updated_slot_words[slot])
            domain = updated_slot_words[slot]
            _direction, _start_idx, length = slot

            # Intersect the current domain with each fixed-letter constraint set.
            # Prefer intersecting with the smaller set first to reduce work.
            empty: set[str] = set()
            constraint_sets: list[set[str]] = [
                word_map.get((length, pos, ch), empty) for pos, ch in fixed_positions
            ]
            constraint_sets.sort(key=len)
            for s in constraint_sets:
                if not domain:
                    break
                if len(domain) > len(s):
                    domain = domain.intersection(s)
                else:
                    domain.intersection_update(s)
            updated_slot_words[slot] = domain
            if len(updated_slot_words[slot]) < n_before:
                cache_dirty[slot] = True
            if not updated_slot_words[slot]:
                return {k: set() for k in slot_words.keys()}

        filled_letters = filled_letters_by_slot.get(slot, Counter())
        n_before = len(updated_slot_words[slot])
        updated_slot_words[slot] = {
            w
            for w in updated_slot_words[slot]
            if is_playable(get_word_counter(w) - filled_letters, available_tiles)
        }
        if len(updated_slot_words[slot]) < n_before:
            cache_dirty[slot] = True
        if not updated_slot_words[slot]:
            return {k: set() for k in slot_words.keys()}

    # Build directed arcs (X -> Y) for each crossing constraint.
    # If X's domain shrinks, only arcs (Z -> X) need to be revisited.
    directed_arcs: list[tuple[Slot, Slot, int, int]] = []
    arcs_by_target: dict[Slot, list[tuple[Slot, Slot, int, int]]] = {}

    for slot_a, slot_b, pos_in_a, pos_in_b in intersection_specs:
        arc_ab = (slot_a, slot_b, pos_in_a, pos_in_b)
        arc_ba = (slot_b, slot_a, pos_in_b, pos_in_a)
        directed_arcs.append(arc_ab)
        directed_arcs.append(arc_ba)
        arcs_by_target.setdefault(slot_b, []).append(arc_ab)
        arcs_by_target.setdefault(slot_a, []).append(arc_ba)

    if solver_config.deterministic:
        directed_arcs.sort()

    def _revise(slot_x: Slot, slot_y: Slot, pos_x: int, pos_y: int) -> bool:
        letters_y = _letters_at_pos(slot_y, pos_y)
        if not letters_y:
            return True

        letters_x = _letters_at_pos(slot_x, pos_x)
        if letters_x.issubset(letters_y):
            return False

        n_before = len(updated_slot_words[slot_x])
        updated_slot_words[slot_x] = {
            w for w in updated_slot_words[slot_x] if w[pos_x] in letters_y
        }
        if len(updated_slot_words[slot_x]) < n_before:
            cache_dirty[slot_x] = True
            return True
        return False

    # AC-3 queue: process only arcs affected by actual domain reductions.
    queue = deque(directed_arcs)
    while queue:
        slot_x, slot_y, pos_x, pos_y = queue.popleft()
        if slot_x not in updated_slot_words or slot_y not in updated_slot_words:
            continue

        if _revise(slot_x, slot_y, pos_x, pos_y):
            if not updated_slot_words[slot_x]:
                return {k: set() for k in slot_words.keys()}

            for arc in arcs_by_target.get(slot_x, []):
                # arc is (Z -> X); skip re-adding the arc we just came from.
                if arc[0] == slot_y:
                    continue
                queue.append(arc)

    return updated_slot_words


def report_progress(
    worker_state: WorkerState,
    board: Board,
    slot_words: SlotWords,
    available_tiles: Counter[str],
) -> None:
    """Report progress of the worker based on filled cells threshold.

    Args:
        worker_state (WorkerState): The state of the worker.
        board (Board): The current state of the board.
        slot_words (SlotWords): Mapping of slots to a domain of possible words.
        available_tiles (Counter[str]): Counter of available tiles.
    """
    n_filled_cells = sum(1 for cell in board.data if cell not in {".", "#"})
    completion_fraction = n_filled_cells / worker_state.total_fillable_cells

    # Check if we have crossed the reporting threshold and our previous best result
    old_threshold = worker_state.last_threshold_reported
    worker_state.last_threshold_reported = max(
        worker_state.last_threshold_reported,
        completion_fraction,
    )
    if completion_fraction < max(
        solver_config.show_progress_threshold,
        # Since our biggest board is 100 cells, this effectively corresponds to a single
        # extra filled cell.  Note `last_threshold_reported` is reset for each new anchor word.
        old_threshold + 0.01,
    ):
        return  # Do not report progress yet

    # Print the full board state in 2D and a special message if solved or almost solved
    if completion_fraction == 1.0:
        print("***    SOLVED!!   ***", flush=True)
    elif completion_fraction >= solver_config.almost_solved_treshold:
        print("*** ALMOST THERE! ***", flush=True)
    board.print(two_d=completion_fraction >= solver_config.almost_solved_treshold)

    # Print a concise progress line
    time_elapsed = time() - worker_state.start_time
    minutes, seconds = divmod(time_elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    time_str = f"{int(hours)}:{int(minutes)}:{int(seconds)}"

    anchor_word = "".join(board[row, solver_config.anchor_col] for row in range(board.n_rows))

    slot_info = ""
    if slot_words:
        total_domain_size = sum(len(words) for words in slot_words.values())
        n_slots = len(slot_words)
        min_domain_size = min(len(words) for words in slot_words.values())
        max_domain_size = max(len(words) for words in slot_words.values())

        def manhattan_dist(slot: Slot) -> int:
            """Get the distance from [0, anchor_col] for a slot."""
            _dir, start_idx, _len = slot
            row, col = board.get_2d_idx(start_idx)
            return row + abs(col - solver_config.anchor_col)

        # Since slots are processed by increasing distance from anchor,
        # the minimum distance among remaining slots indicates progress
        min_distance = min(manhattan_dist(slot) for slot in slot_words.keys())

        # Are there forced moves for the current board state?
        n_forced_moves = sum(1 for words in slot_words.values() if len(words) == 1)

        slot_info = (
            f"*{n_filled_cells}* "
            f"{''.join(available_tiles.elements())} "
            f"W{worker_state.worker_idx}:{anchor_word} "
            f"B:{worker_state.n_boards_examined} "
            f"T:{time_str} "
            f"#S:{n_slots} "
            f"sumD:{total_domain_size} "
            f"minD:{min_domain_size} "
            f"maxD:{max_domain_size} "
            f"minDist:{min_distance} "
            f"F:{n_forced_moves}"
        )
    else:
        slot_info = f"[SOLVED] Time:{time_str}"

    print(slot_info, flush=True)
