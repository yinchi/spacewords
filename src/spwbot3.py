# --- DEFAULT PUZZLE CONFIGURATION ---
# Default puzzle configuration (can be overridden by command-line arguments)
# ⚠️ Only the L-A-S-T PUZZLE_CONFIG is used.  Check carefully. ⚠️
# Format: (category, puzzle_id, (height, width), board_template)
PUZZLE_CONFIG = (
    "Monthly",
    "2512",
    (10, 10),
    """
S Q U A L I D I T Y
. # . . . . . . . #
. . . # . . . . . .
. . . . . # . . . .
. . . . # . . . # .
. . . . . . . . . .
. . . . . . . . # .
. . . . . . # . . .
. . # . . . . . . .
. . . . . . . # . .
""",
)  # 88L (Best general test case - Solution in ~10 mins)
PUZZLE_CONFIG = (
    "Weekly",
    "251207",
    (7, 10),
    """
S U B J E C T I F Y
. . . # . . . . . #
. . . . # . . . . .
. . . . . . . . # .
. . . . . . # . . .
. . # . . . . . . .
. . . . . # . . . .
""",
)
PUZZLE_CONFIG = (
    "Weekly",
    "251214",
    (8, 9),
    """
S E Q U A C I T Y
. . # . . . . . #
. . . . . # . . .
. # . . . . . # .
. . . # . . . . .
. . . . . . # . .
. . . . . # . . .
. . . . # . . . .
""",
)
PUZZLE_CONFIG = (
    "Daily",
    "251215",
    (3, 8),
    """
. . . . # . . .
. . # . . . . .
. . . . . # . .
""",
)

import argparse
import sys

# --- PUZZLE_CONFIG override by command-line arguments ---
parser = argparse.ArgumentParser(description="Anchor-driven parallel crossword solver")
parser.add_argument(
    "--category", type=str, help="Puzzle category (e.g. Weekly, Monthly, Daily)"
)
parser.add_argument("--id", type=str, help="Puzzle ID (e.g. 251207)")
parser.add_argument("--height", type=int, help="Board height")
parser.add_argument("--width", type=int, help="Board width")
parser.add_argument(
    "--board",
    type=str,
    help="Board template as a single string (use . for blanks, # for blocks, no spaces)",
)
args, unknown = parser.parse_known_args()

if args.category and args.id and args.height and args.width and args.board:
    PUZZLE_CONFIG = (args.category, args.id, (args.height, args.width), args.board)

import sys
import time
from multiprocessing import Pool, cpu_count

try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = None  # Gracefully handle if not installed

# --- Solver Parameters ---
DETERMINISTIC = True  # Enable deterministic execution (reproducible results, a bit slower on average)
MAX_WORKERS = None  # Max number of worker processes (None = use all available cores)

# --- Reporting Parameters ---
SHOW_PROGRESS_THRESHOLD = (
    0.90  # Show progress when board is at least this fraction complete (0.0 to 1.0)
)
VERY_CLOSE_THRESHOLD = (
    0.95  # Special notification when board is at least this fraction complete
)
PROGRESS_REPORT_INTERVAL = (
    1000  # Report progress every N boards processed (0 to disable)
)

# --- Anchor-Driven Solver Parameters ---
USE_ANCHOR_PARALLELIZATION = True  # Use anchor column strategy for parallelization
MAX_ANCHOR_CANDIDATES = None  # Maximum number of anchor words to try (None = no limit)
ANCHOR_COLUMN = 0  # Which column to use as anchor (0 = leftmost)
SORT_ANCHORS_BY_PATTERN_SCORE = (
    True  # Sort anchors by row pattern score (highest first)
)


# ============================================================================
# PUZZLE DATABASE
# ============================================================================

PUZZLES = {
    "Daily": {
        "251102": "AAACEEEIKNNOOPPQRSTUV",
        "251122": "ADDEEEEGGHIJNOOOQSSUW",
        "251215": "AAAABDEEGIIJLLOQRTUWY",
    },
    "Weekly": {
        "250817": "AAAAABBCCDDEEEEEEFFGGGHHHIIJKLLLLMMMNNOOOOOOPPPRRRSSTTUUUWYYYYZ",
        "250824": "AAAAAABBCCCDDEEEEEEFFGGGGGHHIIIIJKMMMNNNNOOOOPPPQRRRRSTTTTUUYYZ",
        "250831": "AAAABBBBCCDDEEEEEEEEFFFGGGHHIIIIIIJKLMMMNOOOPPRRRSSSSTTTTUWYYYZ",
        "250907": "AAAAABBCCDDEEEEEFFFGGGGGHHHIIIIKLLMMNNNOOOOOOPPQRRRSTTTTUUVXYYY",
        "250914": "AAAAABBBCCCDDEEEEEEFFGGGGGGHHIIIIIIKLLMMNNNNNNOOPPQRSSTTUUUXYYZ",
        "250921": "AAAABBBCCDDEEEEFFFGGGGHHIIIIIJKLLLMMNNNOOOOOOPPPRSSSTTTUUUVXYYY",
        "250928": "AAAABBBCCCDEEEEEEFFGGGHHHIIIIJKLLMMNNNNOOOOOOPPRRSSSTTTUUVWYYYY",
        "251005": "AAAAABBCCCCDDEEEEEEFFFGGGGGHHIIIKLLMMNOOOOOPPRRRSSSSTTTUUUWXYYZ",
        "251012": "AAAAAAABBBCCDDEEEEEEFFFGGGHHHIIIIKLLMMNNNOOOPPPQRRSSSTTTUUWYYYZ",
        "251019": "AAAAABBCCDDEEEEEEEFFFFGGGHHIIIIIKLLMMMNNOOOOPPQRRRSSSTTTUVXYYYY",
        "251026": "AAAABBBCCCDDDDEEEEFFGGGGHHIIIIIJKLLLMMNNNNOOOOOOPPPRRRTUUUVWYYY",
        "251102": "AAAAABBBCCDDDEEEEEFFFGGGHHHIIIIIIKLMMMNNNOOOPPQRRRRSSSTUUUWXYYY",
        "251109": "AAAABBCCCDDEEEEEEEFFGGGGHHHHIIIJLLLMMNNOOOOPPPRRRSSTTTUUUUVWYYZ",
        "251116": "AAABBCCDEEEEEEFFFGGGGGHHIIIIIJLMMNNNNOOOOOPPPRRSSSSSTTUUUVWYYYZ",
        "251123": "AAAAAABBCCDDEEEEEEEEFFGGGGHHHIIJKLLLMMMMNNNOOOPPRRSSTTTUUUWYYYZ",
        "251130": "AAAAAABBBCCCEEEEEEEFFGGGGHHIIIJLLLMMMMNNOOOOOPPRRRRRSSSTTUVWYYZ",
        "251207": "AAAABBBBCCDDEEEEEEEFFFFGGGHHIIIIIJKLLLMMNNNOOOOPPPQSSSSTTTUUWYY",
        "251214": "AAAAABBCCCCDDEEEEEEEFFFGGGHHHIIIILLLLMMMNNOOOPPQRSSSSTTUUUVWXYY",
    },
    "Monthly": {
        "2510": "AAAABBBBBBBCCCCCDDDEEEEEEEEEEEEEEFFFGGGGGGHHHIIIIIJJLLLLMMMMNNNNNNNOOOOOOOPPPPRRRRSSTTTTTUUUUUWYYYYZ",
        "2511": "AAAAAAAABBBBCCCCCDDDDEEEEEEEEEEFFFFGGGGGGGHHHIIIIIIIIKLLMMMMMMNNNNNNOOOOOOPPPPRRRSSSTTTTTTTUUUVXXYYY",
        "2512": "AAAAAAABBBBBCCCCCDDDEEEEEEEEEEEEEFFGGGGGGGGHHHIIIIJKLLLMMNNNOOOOOOOOPPPPPQRRRRRRRSSSTTTTTTUUUWYYYYYY",
    },
}

# --- Dictionary File Path ---
DICT_PATH = "spw-dict.txt"


def get_puzzle_string(category: str, puzzle_id: str) -> str:
    """Get the letter bag for a specific puzzle."""
    try:
        return PUZZLES[category][puzzle_id]
    except KeyError:
        raise ValueError(
            f"Unknown puzzle for category '{category}' and id '{puzzle_id}'"
        )


def parse_board_string(board_str):
    """Parse a multi-line board string into a clean board representation."""
    return board_str.replace("\n", "").replace(" ", "").upper()


# ============================================================================
# SOLVER FUNCTIONS
# ============================================================================


# update bag if word can be used
def update_letter_bag(current_bag, word):
    for letter in word:
        if letter in current_bag:
            current_bag = current_bag.replace(letter, "", 1)
        else:
            return None
    return current_bag


# confirm that no island exists in board b
def has_no_disconnected_islands(b):
    # Use worker globals if in worker process, otherwise use main globals
    bh = worker_board_height if worker_board_height is not None else board_height
    bw = worker_board_width if worker_board_width is not None else board_width

    code = []
    for i in range(bh * bw):
        code.append(99)
    gp = 0  # group no
    for i in range(bh * bw):
        if b[i] != "#":
            code[i] = gp
            gp += 1
    lastsum = 0
    while sum(code) != lastsum:
        lastsum = sum(code)
        for i in range(bh * bw):
            if b[i] != "#":
                row = int(i / bw)
                col = i % bw
                if row > 0:
                    code[i] = min(code[i], code[i - bw])
                if row < bh - 1:
                    code[i] = min(code[i], code[i + bw])
                if col > 0:
                    code[i] = min(code[i], code[i - 1])
                if col < bw - 1:
                    code[i] = min(code[i], code[i + 1])
                if code[i] == 99:
                    code[i] = gp
                    gp += 1
    if set(code) == {0, 99}:
        return True
    else:
        return False


# ============================================================================
# CONSTRAINT PROPAGATION SOLVER
# ============================================================================

# Global dictionary mapping slot -> board indices (precomputed once)
slot_indices = {}


def build_slot_indices(board):
    """
    Precompute board indices for each slot.
    Maps slot (direction, start_idx, length) -> list of board indices.
    """
    global slot_indices
    slot_indices = {}

    # Use worker globals if in worker process, otherwise use main globals
    bw = worker_board_width if worker_board_width is not None else board_width

    word_positions = get_word_positions_list(board)
    for slot in word_positions:
        direction, start_idx, length = slot

        # Calculate board indices for this slot
        if direction == 0:  # horizontal
            indices = list(range(start_idx, start_idx + length))
        else:  # vertical
            indices = list(range(start_idx, start_idx + length * bw, bw))

        slot_indices[slot] = indices

    return slot_indices


def extract_pattern(board, slot):
    """
    Extract the pattern string for a slot from the board.
    Uses precomputed slot_indices for efficiency.
    """
    # Use worker globals if in worker process, otherwise use main globals
    indices = (
        worker_slot_indices.get(slot)
        if worker_slot_indices is not None
        else slot_indices[slot]
    )
    return "".join(board[i] for i in indices)


def initialize_slot_words(board, letter_bag_param):
    """
    Build initial domains for all word slots.
    Returns a dictionary mapping slot -> list of valid words.
    """
    slot_words = {}
    word_positions = get_word_positions_list(board)

    for slot in word_positions:
        # Get current pattern
        pattern = extract_pattern(board, slot)

        # Find all matching words
        matching = find_words_matching_pattern(pattern)

        # Pre-compute filled letters (avoid repeated calculation)
        filled_letters = pattern.replace(".", "")

        # Filter by letter bag - only keep words that can be formed
        valid_words = []
        for word in matching:
            # Calculate letters needed (excluding already filled letters)
            temp = word
            for ch in filled_letters:
                temp = temp.replace(ch, "", 1)
            # Check if remaining letters are available in bag
            if update_letter_bag(letter_bag_param, temp) is not None:
                valid_words.append(word)

        # Sort only if deterministic behavior required (slower)
        slot_words[slot] = sorted(valid_words) if DETERMINISTIC else valid_words

    return slot_words


def constraint_propagation(board, slot_words, unused_letters):
    """
    Apply constraint propagation to prune domains.

    Removes words that:
    1. Don't match the current pattern on the board
    2. Can't be formed with remaining letters
    3. Would cause global letter constraint violations

    Returns: (board, slot_words, unused_letters) or (None, None, None) if impossible
    """
    if unused_letters is None:
        return None, None, None

    changed = True
    iterations = 0
    max_iterations = 100  # Safety limit

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Sort slots only if deterministic behavior required (slower)
        slots_to_check = (
            sorted(slot_words.keys()) if DETERMINISTIC else list(slot_words.keys())
        )
        for slot in slots_to_check:
            if slot not in slot_words:  # May have been removed
                continue

            words = slot_words[slot]

            if len(words) == 0:
                return None, None, None  # Dead end - fail fast!

            pattern = extract_pattern(board, slot)

            # Pre-calculate letters already on board (avoid repeated calculation)
            filled_letters = pattern.replace(".", "")

            # Filter words from current domain using inline pattern matching
            new_words = []
            for word in words:
                # Check if word matches pattern (inline character-by-character comparison)
                matches = True
                for i, ch in enumerate(pattern):
                    if ch != "." and ch != word[i]:
                        matches = False
                        break

                if not matches:
                    continue

                # Check if word can be formed with remaining letters
                # Calculate only NEW letters needed (word minus already placed letters)
                temp = word
                for ch in filled_letters:
                    temp = temp.replace(ch, "", 1)

                if update_letter_bag(unused_letters, temp) is not None:
                    new_words.append(word)

            # Update domain if it changed
            if len(new_words) < len(words):
                slot_words[slot] = new_words
                changed = True

                # If domain reduced to 0, fail
                if len(new_words) == 0:
                    return None, None, None

        # --- ARC CONSISTENCY: Prune by set intersection at crossing cells ---
        # For each pair of crossing slots, enforce that their possible letters at the crossing cell are consistent
        # slot_indices: slot -> list of board indices
        slot_list = list(slot_words.keys())
        for i, slot1 in enumerate(slot_list):
            indices1 = (
                worker_slot_indices.get(slot1)
                if worker_slot_indices is not None
                else slot_indices[slot1]
            )
            for j, idx1 in enumerate(indices1):
                # For each cell in slot1, check if another slot covers it
                for slot2 in slot_list:
                    if slot2 == slot1:
                        continue
                    indices2 = (
                        worker_slot_indices.get(slot2)
                        if worker_slot_indices is not None
                        else slot_indices[slot2]
                    )
                    if idx1 in indices2:
                        k = indices2.index(idx1)
                        # Now, slot1[j] and slot2[k] are the same cell
                        # Get possible letters at this cell for both slots
                        words1 = slot_words[slot1]
                        words2 = slot_words[slot2]
                        if not words1 or not words2:
                            continue
                        letters1 = set(w[j] for w in words1)
                        letters2 = set(w[k] for w in words2)
                        allowed_letters = letters1 & letters2
                        # Prune words in slot1 and slot2 that don't have allowed letters at this cell
                        new_words1 = [w for w in words1 if w[j] in allowed_letters]
                        new_words2 = [w for w in words2 if w[k] in allowed_letters]
                        if len(new_words1) < len(words1):
                            slot_words[slot1] = new_words1
                            changed = True
                            if len(new_words1) == 0:
                                return None, None, None
                        if len(new_words2) < len(words2):
                            slot_words[slot2] = new_words2
                            changed = True
                            if len(new_words2) == 0:
                                return None, None, None

    return board, slot_words, unused_letters


# Global variables for worker processes (set by init_worker)
worker_board_width = None
worker_board_height = None
worker_word_set = None
worker_slot_indices = None
worker_letter_frequencies = None
worker_word_dictionary = None
worker_mode = False  # Set to True in worker processes
worker_id = 0  # Worker identifier (0 for main process)

# Shared counter for assigning sequential worker IDs
_worker_counter = None


def init_worker(bw, bh, ws, si, lf, wd, tfc, st, worker_counter):
    """
    Initialize worker process with shared data.
    Called once per worker process at pool creation.
    """
    global worker_board_width, worker_board_height, worker_word_set
    global \
        worker_slot_indices, \
        worker_letter_frequencies, \
        worker_word_dictionary, \
        worker_mode
    global \
        boards_processed_count, \
        last_threshold_reported_at, \
        total_fillable_cells, \
        start_time, \
        worker_id
    global _worker_counter

    import os

    worker_board_width = bw
    worker_board_height = bh
    worker_word_set = ws
    worker_slot_indices = si
    worker_letter_frequencies = lf
    worker_word_dictionary = wd
    worker_mode = True  # Mark as worker process

    # Assign sequential worker ID using shared counter
    _worker_counter = worker_counter
    with _worker_counter.get_lock():
        worker_id = _worker_counter.value
        _worker_counter.value += 1

    pid = os.getpid()
    print(f"Worker {worker_id}: PID {pid}")
    sys.stdout.flush()

    # Initialize progress tracking globals for worker
    boards_processed_count = 0
    last_threshold_reported_at = -1
    total_fillable_cells = tfc
    start_time = st


def solve_with_cp(board, slot_words, unused_letters, depth=0):
    """
    Main constraint propagation solver.

    Uses:
    - Constraint propagation to prune impossible words
    - Forced move detection (slots with only 1 valid word)
    - Minimum Remaining Values (MRV) heuristic for slot selection
    - Backtracking when needed
    """

    # Progress tracking (works in both main and worker processes)
    global boards_processed_count
    boards_processed_count += 1

    # Debug: Report depth changes
    if depth > 0 and boards_processed_count % 1000 == 0:
        worker_prefix = f"Worker {worker_id}: " if worker_mode else "Main: "
        # print(f"DEBUG {worker_prefix}depth={depth}, slots={len(slot_words)}, boards={boards_processed_count}")
        sys.stdout.flush()

    # Periodic progress report (show immediately after incrementing counter)
    report_progress(board, slot_words, unused_letters)

    # Apply constraint propagation
    # Use shallow copy + copy-on-write for efficiency
    slot_words_copy = {slot: words[:] for slot, words in slot_words.items()}
    board, slot_words_copy, unused_letters = constraint_propagation(
        board, slot_words_copy, unused_letters
    )

    if slot_words_copy is None:
        # if depth == 0:
        #     print(f"DEBUG: Constraint propagation FAILED at initial call!")
        #     print(f"  This means the puzzle has no solution or initialization is incorrect.")
        return []  # Failed - backtrack

    slot_words = slot_words_copy  # Use the propagated domains

    # Apply forced moves (slots with only 1 valid word)
    global last_threshold_reported_at
    forced_moves_count = 0
    while True:
        forced = [slot for slot in slot_words if len(slot_words[slot]) == 1]
        if not forced:
            break

        # Place all forced words
        for slot in forced:
            word = slot_words[slot][0]
            direction, start_idx, length = slot

            # Get pattern BEFORE placing word (to know which letters are already there)
            old_pattern = extract_pattern(board, slot)

            # Calculate NEW letters needed (word minus already placed letters)
            letters_to_remove = word
            for ch in old_pattern.replace(".", ""):
                if ch in letters_to_remove:
                    letters_to_remove = letters_to_remove.replace(ch, "", 1)

            # Update unused letters BEFORE placing word
            unused_letters = update_letter_bag(unused_letters, letters_to_remove)

            if unused_letters is None:
                return []  # Impossible - can't form this word with available letters

            # Now place word on board
            board = add_word_to_board(board, word, start_idx, direction)

            # Validate crossing words after placement
            if not validate_crossing_words(board, word, start_idx, direction):
                return []  # Invalid crossing words created

            # Remove this slot from domains
            del slot_words[slot]
            forced_moves_count += 1

            # Check if threshold met after placing this forced move
            report_if_threshold_met(board, unused_letters, slot_words)

        # Propagate constraints again after forced moves
        board, slot_words, unused_letters = constraint_propagation(
            board, slot_words, unused_letters
        )

        if slot_words is None:
            return []

    # Check if threshold met for detailed reporting (only when progress is made)
    # Show in both main and workers
    report_if_threshold_met(board, unused_letters, slot_words)

    # Check if solved
    if len(slot_words) == 0:
        if is_valid_board(board):
            # Solution found! Print immediately for visibility
            if worker_mode:
                print(f"\n*** SOLUTION FOUND by worker {worker_id}! ***")
            else:
                print(f"\n*** SOLUTION FOUND by main process! ***")
            sys.stdout.flush()
            return board
        else:
            # Board is complete but has invalid words - this shouldn't happen
            print(f"\nWARNING: Complete board but is_valid_board() failed!")
            if worker_mode:
                print(f"  Worker {worker_id}")
            display_board(board)
            sys.stdout.flush()
        return []

    # Check for dead slots (no valid words)
    if any(len(words) == 0 for words in slot_words.values()):
        return []

    # Choose slot by radiating from anchor (top-left corner)
    # PRIMARY: Manhattan distance (prioritize cells near anchor for maximum constraint propagation)
    # SECONDARY: Domain size (MRV heuristic - prefer constrained slots)
    # TERTIARY: Direction (prefer horizontal)
    def slot_priority(slot):
        direction, start_idx, length = slot
        domain_size = len(slot_words[slot])
        # Calculate distance from top-left corner (0,0) where anchor is located
        bw = worker_board_width if worker_board_width is not None else board_width
        row = start_idx // bw
        col = start_idx % bw
        # Distance metric: row + col (Manhattan distance from anchor)
        distance = row + col
        # PRIMARY sort by distance (radiate from anchor), SECONDARY by domain size, TERTIARY by direction
        return (distance, domain_size, direction, slot)

    if DETERMINISTIC:
        # Sort by priority for deterministic selection
        min_slot = min(sorted(slot_words.keys()), key=slot_priority)
    else:
        # Fast selection with distance-first priority
        min_slot = min(slot_words.keys(), key=slot_priority)
    words_to_try = slot_words[min_slot]

    # Sort words by letter frequency (try rare letters first to fail fast)
    words_to_try = sort_words_by_frequency(words_to_try)

    # Sequential execution
    for word in words_to_try:
        direction, start_idx, length = min_slot

        # Place word on board
        new_board = add_word_to_board(board, word, start_idx, direction)

        # Validate crossing words immediately after placement
        if not validate_crossing_words(new_board, word, start_idx, direction):
            continue  # Skip this word - creates invalid crossings

        # Calculate letters needed for this word
        letters_to_remove = word
        pattern = extract_pattern(board, min_slot)
        for ch in pattern.replace(".", ""):
            if ch in letters_to_remove:
                letters_to_remove = letters_to_remove.replace(ch, "", 1)

        new_unused = update_letter_bag(unused_letters, letters_to_remove)

        if new_unused is None:
            continue  # This word uses unavailable letters

        # Create new domains without the anchor slot (it's now committed, like forced moves)
        new_slot_words = {
            slot: words[:] for slot, words in slot_words.items() if slot != min_slot
        }

        # Recurse
        result = solve_with_cp(new_board, new_slot_words, new_unused, depth + 1)
        if result != []:
            return result

    return []  # All words failed - backtrack


# add w to board b at idx i, dir=0 for hori
def add_word_to_board(board, word, start_index, direction):
    # Use worker globals if in worker process
    bw = worker_board_width if worker_board_width is not None else board_width

    board_list = list(board)
    if direction == 0:
        for j in range(start_index, start_index + len(word)):
            board_list[j] = word[j - start_index]
        return "".join(board_list)
    else:
        for j in range(start_index, start_index + len(word) * bw, bw):
            board_list[j] = word[int((j - start_index) / bw)]
        return "".join(board_list)


# Helper for parallel execution that takes board_width as parameter
def add_word_to_board_with_width(board, word, start_index, direction, bw):
    board_list = list(board)
    if direction == 0:
        for j in range(start_index, start_index + len(word)):
            board_list[j] = word[j - start_index]
        return "".join(board_list)
    else:
        for j in range(start_index, start_index + len(word) * bw, bw):
            board_list[j] = word[int((j - start_index) / bw)]
        return "".join(board_list)


# display board
def display_board(board):
    # Use worker globals if in worker process
    bh = worker_board_height if worker_board_height is not None else board_height
    bw = worker_board_width if worker_board_width is not None else board_width

    for row in range(bh):
        display_string = ""
        for col in range(bw):
            if board[row * bw + col] != "#":
                display_string += " " + board[row * bw + col]
            else:
                display_string += " #"
        print(display_string)


# count nb of letters filled in board
def count_filled_letters(board):
    return len(board.replace("#", "").replace(".", ""))


def report_if_threshold_met(board, unused_letters, slot_words):
    """
    Check if board completion meets threshold and report if needed.
    Updates global last_threshold_reported_at to avoid duplicate reports.
    """
    global last_threshold_reported_at
    filled_letters = count_filled_letters(board)
    completion_fraction = (
        (filled_letters / total_fillable_cells) if total_fillable_cells > 0 else 0
    )
    if (
        completion_fraction >= SHOW_PROGRESS_THRESHOLD
        and filled_letters > last_threshold_reported_at
    ):
        report_threshold_met(board, unused_letters, slot_words)
        last_threshold_reported_at = filled_letters


def format_elapsed_time(elapsed_seconds):
    """
    Format elapsed time as HH:MM:SS.s or MM:SS.s
    """
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    if hours > 0:
        return f"{hours}h:{minutes:02d}:{seconds:02.0f}"
    else:
        return f"{minutes:02d}:{seconds:04.1f}"


def report_progress(board, slot_words=None, unused_letters=None):
    """
    Display periodic progress report (every PROGRESS_REPORT_INTERVAL boards).
    Shows filled count, board state, boards processed, elapsed time, and domain size.
    """
    if (
        PROGRESS_REPORT_INTERVAL > 0
        and boards_processed_count % PROGRESS_REPORT_INTERVAL == 0
    ):
        filled = count_filled_letters(board)
        elapsed = time.time() - start_time
        time_str = format_elapsed_time(elapsed)
        # Extract anchor column word (column ANCHOR_COLUMN)
        anchor_col = ANCHOR_COLUMN if "ANCHOR_COLUMN" in globals() else 0
        bh = worker_board_height if worker_board_height is not None else board_height
        bw = worker_board_width if worker_board_width is not None else board_width
        anchor_word = "".join(board[anchor_col + bw * row] for row in range(bh))

        # Format boards processed with worker ID
        boards_str = (
            f"{worker_id}~{anchor_word}~{boards_processed_count}"
            if worker_mode
            else str(boards_processed_count)
        )

        # Prepare slot info and anchor word
        slot_info = ""
        if slot_words is not None:
            num_slots = len(slot_words)
            if num_slots > 0:
                total_domain = sum(len(words) for words in slot_words.values())
                min_domain = min(len(words) for words in slot_words.values())
                max_domain = max(len(words) for words in slot_words.values())
                manhattan_distance = min(
                    (slot[1] // bw) + (slot[1] % bw) for slot in slot_words.keys()
                )
                forced_moves = sum(
                    1 for words in slot_words.values() if len(words) == 1
                )
                slot_info = f"s:{num_slots} Md:{manhattan_distance} d:{total_domain}[{min_domain},{max_domain}]"
                if forced_moves > 0:
                    slot_info += f" fm:{forced_moves}"
            else:
                slot_info = "[SOLVED]"
        # Compose and print the progress line
        print(f"*{filled}*  {board}  {boards_str}  {time_str}  {slot_info}")
        sys.stdout.flush()  # Ensure output appears immediately


def report_threshold_met(board, unused_letters, slot_words=None):
    """
    Display detailed progress report when board completion meets threshold criteria.
    Shows formatted board, unused letters, progress stats, and domain info.
    Used by both backtracking and CP solvers.
    """
    # Use worker globals if in worker process for total_fillable_cells calculation
    tfc = total_fillable_cells

    # Calculate completion fraction (total_fillable_cells is computed once at start)
    filled_letters = count_filled_letters(board)
    completion_fraction = (filled_letters / tfc) if tfc > 0 else 0

    if completion_fraction >= SHOW_PROGRESS_THRESHOLD:
        if completion_fraction >= VERY_CLOSE_THRESHOLD:
            if completion_fraction == 1.0:
                print("***    SOLVED!!   ***")
            else:
                print("*** ALMOST THERE! ***")
        display_board(board)

        # Format elapsed time
        elapsed = time.time() - start_time
        time_str = format_elapsed_time(elapsed)

        # Extract anchor column word (column ANCHOR_COLUMN)
        anchor_col = ANCHOR_COLUMN if "ANCHOR_COLUMN" in globals() else 0
        bh = worker_board_height if worker_board_height is not None else board_height
        bw = worker_board_width if worker_board_width is not None else board_width
        anchor_word = "".join(board[anchor_col + bw * row] for row in range(bh))

        # Format boards processed with worker ID
        boards_str = (
            f"{worker_id}~{anchor_word}~{boards_processed_count}"
            if worker_mode
            else str(boards_processed_count)
        )

        # Prepare slot info and anchor word
        slot_info = ""
        if slot_words is not None:
            num_slots = len(slot_words)
            if num_slots > 0:
                total_domain = sum(len(words) for words in slot_words.values())
                min_domain = min(len(words) for words in slot_words.values())
                max_domain = max(len(words) for words in slot_words.values())
                manhattan_distance = min(
                    (slot[1] // bw) + (slot[1] % bw) for slot in slot_words.keys()
                )
                forced_moves = sum(
                    1 for words in slot_words.values() if len(words) == 1
                )
                slot_info = f"s:{num_slots} Md:{manhattan_distance} d:{total_domain}[{min_domain},{max_domain}]"
                if forced_moves > 0:
                    slot_info += f" fm:{forced_moves}"
            else:
                slot_info = "[SOLVED]"
        # Compose and print the progress line
        print(
            f"*{filled_letters}*  {unused_letters}  {boards_str}  {time_str}  {slot_info}"
        )
        print("----")


def validate_crossing_words(board, word, start_idx, direction):
    """
    Validate only the crossing words affected by placing a word at the given position.
    Much faster than validating the entire board.

    Args:
        board: Current board state
        word: The word being placed
        start_idx: Starting index of the word
        direction: 0 for horizontal, 1 for vertical

    Returns:
        True if all crossing words are valid, False otherwise
    """
    # Use worker globals if in worker process
    bw = worker_board_width if worker_board_width is not None else board_width
    bh = worker_board_height if worker_board_height is not None else board_height
    ws = worker_word_set if worker_word_set is not None else word_set

    if direction == 0:  # Horizontal word - check vertical crossings
        for i, letter in enumerate(word):
            pos = start_idx + i
            col = pos % bw

            # Extract vertical word containing this position
            vertical_word = ""
            for row in range(bh):
                vertical_word += board[col + bw * row]

            # Check all complete vertical words in this column
            vertical_segments = vertical_word.split("#")
            for segment in vertical_segments:
                if len(segment) > 1 and "." not in segment and segment not in ws:
                    return False

    else:  # Vertical word - check horizontal crossings
        for i, letter in enumerate(word):
            pos = start_idx + i * bw
            row = pos // bw

            # Extract horizontal word containing this position
            horizontal_word = ""
            for col in range(bw):
                horizontal_word += board[row * bw + col]

            # Check all complete horizontal words in this row
            horizontal_segments = horizontal_word.split("#")
            for segment in horizontal_segments:
                if len(segment) > 1 and "." not in segment and segment not in ws:
                    return False

    return True


# check whether all completed words on the board are words
def is_valid_board(board):
    # Use worker globals if in worker process
    bw = worker_board_width if worker_board_width is not None else board_width
    bh = worker_board_height if worker_board_height is not None else board_height
    ws = worker_word_set if worker_word_set is not None else word_set

    invalid_words = []

    for row in range(bh):
        horizontal_word = ""
        for col in range(bw):
            horizontal_word += board[row * bw + col]
        horizontal_word_segments = horizontal_word.split("#")
        for word in horizontal_word_segments:
            if len(word) > 1 and "." not in word and word not in ws:
                invalid_words.append(f"H row {row}: '{word}'")

    for col in range(bw):
        vertical_word = ""
        for row in range(bh):
            vertical_word += board[col + bw * row]
        vertical_word_segments = vertical_word.split("#")
        for word in vertical_word_segments:
            if len(word) > 1 and "." not in word and word not in ws:
                invalid_words.append(f"V col {col}: '{word}'")

    if invalid_words:
        # Print invalid words for debugging
        print(f"  Invalid words found: {', '.join(invalid_words)}")
        sys.stdout.flush()
        return False

    return True


# return a list of word positions as tuples
# (0,2,4) denotes a 4L horizontal word starting at idx 2
# (1,2,5) denotes a 5L vertical word starting at idx 2
def get_word_positions_list(board):
    # Use worker globals if in worker process, otherwise use main globals
    bh = worker_board_height if worker_board_height is not None else board_height
    bw = worker_board_width if worker_board_width is not None else board_width

    word_positions_list = []
    for row in range(bh):
        horizontal_word = ""
        for col in range(bw):
            horizontal_word += board[row * bw + col]
        horizontal_word_segments = horizontal_word.split("#")
        num_horizontal_segments = len(horizontal_word_segments)
        position_offset = 0
        for segment_index in range(num_horizontal_segments):
            if (
                "." in horizontal_word_segments[segment_index]
                and len(horizontal_word_segments[segment_index]) > 1
            ):
                word_positions_list.append(
                    (
                        0,
                        row * bw + position_offset,
                        len(horizontal_word_segments[segment_index]),
                    )
                )
            position_offset += len(horizontal_word_segments[segment_index]) + 1
    for col in range(bw):
        vertical_word = ""
        for row_idx in range(bh):
            vertical_word += board[col + bw * row_idx]
        vertical_word_segments = vertical_word.split("#")
        num_vertical_segments = len(vertical_word_segments)
        position_offset = 0
        for segment_index in range(num_vertical_segments):
            if (
                "." in vertical_word_segments[segment_index]
                and len(vertical_word_segments[segment_index]) > 1
            ):
                word_positions_list.append(
                    (
                        1,
                        col + bw * position_offset,
                        len(vertical_word_segments[segment_index]),
                    )
                )
            position_offset += len(vertical_word_segments[segment_index]) + 1
    return word_positions_list


# find set of words meeting the s pattern e.g. 'A.B..'
def find_words_matching_pattern(pattern):
    # Use worker globals if in worker process
    wd = (
        worker_word_dictionary
        if worker_word_dictionary is not None
        else word_dictionary
    )

    pattern = pattern.upper()
    letter_positions = []
    for i in range(len(pattern)):
        if pattern[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            letter_positions.append(i)
    if len(letter_positions) > 0:
        if len(letter_positions) % 2 == 1:
            start_index = 1
            result_set = wd.get(
                (len(pattern), letter_positions[0], pattern[letter_positions[0]]), set()
            )
        else:
            start_index = 2
            set_i = wd.get(
                (len(pattern), letter_positions[0], pattern[letter_positions[0]]), set()
            )
            set_j = wd.get(
                (len(pattern), letter_positions[1], pattern[letter_positions[1]]), set()
            )
            result_set = set_i.intersection(set_j)
        for i in range(start_index, len(letter_positions) - 1, 2):
            set_i = wd.get(
                (len(pattern), letter_positions[i], pattern[letter_positions[i]]), set()
            )
            set_j = wd.get(
                (
                    len(pattern),
                    letter_positions[i + 1],
                    pattern[letter_positions[i + 1]],
                ),
                set(),
            )
            result_set = result_set.intersection(set_i.intersection(set_j))
        return result_set
    else:
        result_set = set()
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            result_set = result_set.union(wd.get((len(pattern), 0, letter), set()))
        return result_set


# rearrange words in a list in order of reverse product of letter frequencies
def sort_words_by_frequency(word_list):
    """
    Sort words by rarity (rare letters first).
    Uses product of letter frequencies - lower product = rarer letters.
    Product strongly emphasizes rare letters (Q, Z, X, J) to fail-fast in backtracking.
    A word with even one rare letter will be tried much earlier than common-letter words.
    """
    if not word_list:
        return []

    # Use worker globals if in worker process
    lf = (
        worker_letter_frequencies
        if worker_letter_frequencies is not None
        else letter_frequencies
    )

    def frequency_product(word):
        product = 1
        for letter in word:
            product *= lf[letter]
        return product

    return sorted(word_list, key=frequency_product)


# ============================================================================
# ANCHOR-DRIVEN PARALLEL SOLVER
# ============================================================================


def solve_with_anchor_column_wrapper(args):
    """
    Wrapper to unpack arguments for solve_with_anchor_column.
    Needed because multiprocessing can't pickle lambda functions.
    """
    return solve_with_anchor_column(*args)


def solve_with_anchor_column(anchor_word, board_template, letter_bag_param):
    """
    Solve puzzle with a specific anchor word placed in column 0.
    This function runs in a worker process.

    Args:
        anchor_word: Word to place vertically at column 0
        board_template: Initial board configuration
        letter_bag_param: Available letters

    Returns:
        (worker_id, anchor_word, solution_board, boards_processed)
    """
    global boards_processed_count, worker_id
    boards_processed_count = 0

    # Place anchor word vertically at column 0
    board = list(board_template)

    # Calculate which letters are NEW (not already on the template)
    letters_to_remove = []
    for row_idx, letter in enumerate(anchor_word):
        old_letter = board_template[row_idx * worker_board_width]
        if old_letter == "." or old_letter == "#":
            # This is a new letter being placed
            letters_to_remove.append(letter)
        elif old_letter != letter:
            # Conflict: template has a different letter - invalid anchor
            return (worker_id, anchor_word, [], 0)
        # else: letter already matches, don't consume from bag

        board[row_idx * worker_board_width] = letter
    board = "".join(board)

    # Validate no disconnected islands after placing anchor
    if not has_no_disconnected_islands(board):
        return (worker_id, anchor_word, [], 0)

    # Calculate unused letters (remove only NEW letters)
    unused_letters = letter_bag_param
    for letter in letters_to_remove:
        if letter in unused_letters:
            unused_letters = unused_letters.replace(letter, "", 1)
        else:
            # Anchor word uses letter not in bag - invalid
            return (worker_id, anchor_word, [], 0)

    # Build slot indices for this board configuration
    # Use worker's global slot_indices (will be populated by this call)
    build_slot_indices(board)

    # Initialize domains with anchor constraints already applied
    slot_words = initialize_slot_words(board, unused_letters)

    # Check if any slot has no valid words (fail-fast)
    if any(len(words) == 0 for words in slot_words.values()):
        return (worker_id, anchor_word, [], 0)

    # Calculate domain statistics for progress report
    total_domain = sum(len(words) for words in slot_words.values())
    min_domain = min(len(words) for words in slot_words.values()) if slot_words else 0
    max_domain = max(len(words) for words in slot_words.values()) if slot_words else 0

    print(
        f"Worker {worker_id}: Anchor '{anchor_word}' - {len(slot_words)} slots, domain [{min_domain},{max_domain}], total {total_domain}"
    )
    sys.stdout.flush()

    # Solve with constraint propagation
    solution = solve_with_cp(board, slot_words, unused_letters, depth=0)

    return (worker_id, anchor_word, solution, boards_processed_count)


def calculate_row_score(word_count, word_length):
    """
    Calculate row score for a horizontal word slot based on available word count.

    Formula:
    - If L=2: score = N
    - If L>2: score = N / 2^(L+1)

    where N is the number of words of length L.

    Rationale: Shorter words have exponentially more options, so we penalize
    longer words to favor patterns that create shorter, more flexible slots.
    """
    if word_length == 2:
        return word_count
    else:
        return word_count / (2 ** (word_length + 1))


def calculate_anchor_pattern_score(anchor_word, board_template):
    """
    Calculate pattern score for an anchor column word based on the horizontal
    row patterns it would create.

    For each row, determines the horizontal word length that would cross the
    anchor column, counts available words of that length, and applies the
    row score formula. Returns the sum of all row scores.
    """
    total_score = 0.0

    # For each position in the anchor word
    for row_idx, letter in enumerate(anchor_word):
        # Determine the horizontal word length for this row
        # Start by finding the word slot containing column ANCHOR_COLUMN
        row_start = row_idx * board_width
        row_end = row_start + board_width
        row_pattern = board_template[row_start:row_end]

        # Find all horizontal word segments in this row
        segments = row_pattern.split("#")

        # Find which segment contains our anchor column
        col_position = 0
        for segment in segments:
            segment_len = len(segment)
            if col_position <= ANCHOR_COLUMN < col_position + segment_len:
                # This segment contains our anchor column
                if "." in segment or segment_len > 1:
                    # Build pattern for this word slot with anchor letter placed
                    relative_pos = ANCHOR_COLUMN - col_position
                    pattern_list = list(segment)
                    pattern_list[relative_pos] = letter
                    pattern = "".join(pattern_list)

                    # Count words matching this pattern
                    matching_words = find_words_matching_pattern(pattern)
                    word_count = len(matching_words)
                    word_length = len(pattern)

                    # Calculate row score
                    row_score = calculate_row_score(word_count, word_length)
                    total_score += row_score
                break
            col_position += segment_len + 1  # +1 for the '#' separator

    return total_score


def generate_anchor_candidates(board_template, letter_bag_param, max_candidates=100):
    """
    Generate candidate anchor column words.

    Returns list of valid words that:
    1. Match the length (board_height)
    2. Match any fixed letters in column 0
    3. Can be formed from letter_bag
    4. Don't create disconnected islands
    5. Optionally sorted by pattern score (if SORT_ANCHORS_BY_PATTERN_SCORE is True)
    """
    # Extract column 0 pattern
    column_pattern = ""
    for row in range(board_height):
        column_pattern += board_template[row * board_width]

    print(f"Column {ANCHOR_COLUMN} pattern: '{column_pattern}'")

    # Find matching words
    candidates = find_words_matching_pattern(column_pattern)
    print(f"Found {len(candidates)} words matching pattern")

    # Filter by letter bag availability
    valid_candidates = []
    for word in candidates:
        if update_letter_bag(letter_bag_param, word) is not None:
            # Quick check: placing this word doesn't create islands
            test_board = list(board_template)
            for row_idx, letter in enumerate(word):
                test_board[row_idx * board_width] = letter
            test_board = "".join(test_board)

            if has_no_disconnected_islands(test_board):
                valid_candidates.append(word)

    print(
        f"Found {len(valid_candidates)} valid anchor candidates (after bag and island checks)"
    )

    # Sort by pattern score if enabled
    if SORT_ANCHORS_BY_PATTERN_SCORE and len(valid_candidates) > 0:
        print(f"Calculating pattern scores for {len(valid_candidates)} candidates...")
        candidate_scores = []
        for word in valid_candidates:
            score = calculate_anchor_pattern_score(word, board_template)
            candidate_scores.append((word, score))

        # Sort by score descending (highest first)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        valid_candidates = [word for word, score in candidate_scores]

        print(
            f"Sorted candidates by pattern score (range: [{candidate_scores[-1][1]:.2f}, {candidate_scores[0][1]:.2f}])"
        )

    if max_candidates is not None and len(valid_candidates) > max_candidates:
        print(f"Limiting to first {max_candidates} candidates")
        valid_candidates = valid_candidates[:max_candidates]

    return valid_candidates


def solve_with_parallel_anchors():
    """
    Main solving function using parallel anchor strategy.
    Each worker gets a different anchor column word and solves independently.
    """
    print("\n" + "=" * 80)
    print("ANCHOR-DRIVEN PARALLEL SOLVER")
    print("=" * 80)

    # Generate anchor candidates
    anchor_candidates = generate_anchor_candidates(
        initial_board, letter_bag, max_candidates=MAX_ANCHOR_CANDIDATES
    )

    if len(anchor_candidates) == 0:
        print("ERROR: No valid anchor candidates found!")
        return []

    # Display top candidates
    print(f"\nTop 10 anchor candidates:")
    for i, word in enumerate(anchor_candidates[:10], 1):
        print(f"  {i:2d}. {word}")
    print()

    # Create worker pool if not already created
    if global_worker_pool is None:
        from multiprocessing import Pool, Value

        worker_counter_local = Value("i", 1)
        num_workers = cpu_count() if MAX_WORKERS is None else MAX_WORKERS
        print(f"Creating worker pool with {num_workers} processes...")
        init_args = (
            board_width,
            board_height,
            word_set,
            slot_indices,
            letter_frequencies,
            word_dictionary,
            total_fillable_cells,
            start_time,
            worker_counter_local,
        )
        pool = Pool(processes=num_workers, initializer=init_worker, initargs=init_args)
    else:
        pool = global_worker_pool
        num_workers = cpu_count() if MAX_WORKERS is None else MAX_WORKERS

    print(f"Distributing {len(anchor_candidates)} anchors to {num_workers} workers...")
    print()
    sys.stdout.flush()

    # Calculate unused letters (remove letters already on initial board)
    unused_letters_for_workers = letter_bag
    for i in range(board_height * board_width):
        if initial_board[i] != "." and initial_board[i] != "#":
            unused_letters_for_workers = unused_letters_for_workers.replace(
                initial_board[i], "", 1
            )

    print(
        f"Initial board has {len(letter_bag) - len(unused_letters_for_workers)} letters already placed"
    )
    print(f"Unused letters for solving: {unused_letters_for_workers}")
    print()

    # Create work packages
    work_packages = [
        (word, initial_board, unused_letters_for_workers) for word in anchor_candidates
    ]

    # Process in parallel - first solution wins
    solution_found = False
    final_result = []
    total_worker_boards = 0
    workers_finished = 0

    try:
        for wid, anchor_word, solution, worker_boards in pool.imap_unordered(
            solve_with_anchor_column_wrapper, work_packages
        ):
            total_worker_boards += worker_boards
            workers_finished += 1

            if solution != []:
                # Solution found! Terminate remaining work
                print(f"\n{'=' * 80}")
                print(
                    f"*** SOLUTION FOUND by Worker {wid} with anchor '{anchor_word}'! ***"
                )
                print(f"{'=' * 80}")
                final_result = solution
                solution_found = True
                pool.terminate()
                pool.join()
                break
            else:
                # Show all failures with correct worker ID
                print(
                    f"  Worker {wid}: [{workers_finished}/{len(anchor_candidates)}] Anchor '{anchor_word}' failed after {worker_boards:,} boards"
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        pool.terminate()
        pool.join()
        return []
    finally:
        # Update global boards count
        global boards_processed_count
        boards_processed_count = total_worker_boards

    if solution_found:
        return final_result
    else:
        print(
            f"\nNo solution found with any of the {len(anchor_candidates)} anchor candidates"
        )
        return []


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Module-level variables for board dimensions and puzzle data (set in main)
board_height = None
board_width = None
initial_board = None
letter_bag = None
letter_frequencies = None
word_set = None
word_dictionary = None
slot_indices = {}
start_time = None
total_fillable_cells = None

if __name__ == "__main__":
    # Unpack puzzle configuration
    puzzle_category, puzzle_id, board_dims, board_template = PUZZLE_CONFIG
    board_height, board_width = board_dims

    # Get puzzle letter bag and parse board
    letter_bag = get_puzzle_string(puzzle_category, puzzle_id)
    initial_board = parse_board_string(board_template)
    max_word_length = max(board_width, board_height)

    # Validate board string length matches configured dimensions
    expected_length = board_height * board_width
    if len(initial_board) != expected_length:
        raise ValueError(
            f"Board length mismatch: expected {expected_length} ({board_height}x{board_width}), got {len(initial_board)}"
        )

    # Validate no blocks in column 0
    col0_has_block = any(
        initial_board[row * board_width] == "#" for row in range(board_height)
    )
    if col0_has_block:
        raise ValueError(
            "Invalid puzzle: Column 0 contains a block ('#'). Anchor column must be fully open."
        )

    # Validate no disconnected islands in initial board
    if not has_no_disconnected_islands(initial_board):
        raise ValueError(
            "Initial board has disconnected islands! All white cells must be connected."
        )

    num_blocks = initial_board.count("#")
    print(f"Selected puzzle: {puzzle_category} {puzzle_id}")
    print(f"Board dimensions: {board_height}x{board_width}")
    print(f"Letter bag: {letter_bag}")
    print(f"Available letters: {len(letter_bag)}")
    print(f"Number of blocks: {num_blocks}")
    print()

    # load word list
    words = []
    with open(DICT_PATH) as f:
        for line in f:
            word = line.replace("\n", "")
            if (
                update_letter_bag(letter_bag, word) != None
                and len(word) <= max_word_length
            ):
                words.append(word)
    word_set = set(words)

    # Build inverted index for fast pattern matching:
    # Maps (word_length, position, letter) -> set of words with that letter at that position
    # Example: (5, 2, 'T') maps to all 5-letter words with 'T' at index 2
    # Used by find_words_matching_pattern() to quickly find candidates by intersecting sets
    word_dictionary = {}
    for word in words:
        for position in range(len(word)):
            key = (len(word), position, word[position])
            if key not in word_dictionary:
                word_dictionary[key] = {word}
            else:
                word_dictionary[key].add(word)

    # compile dict of letter frequency in percentage
    letter_frequencies = {}
    total_letters_count = 0
    for word in words:
        total_letters_count += len(word)
        for letter in word:
            if letter in letter_frequencies:
                letter_frequencies[letter] += 1
            else:
                letter_frequencies[letter] = 1
    for letter in letter_frequencies:
        letter_frequencies[letter] /= total_letters_count / 100

    start_time = time.time()
    print("\nSearching for solutions...")
    print("Initial board:")
    display_board(initial_board)
    print()

    boards_processed_count = 0  # number of boards processed
    total_fillable_cells = board_height * board_width - initial_board.count(
        "#"
    )  # constant: total non-block cells
    last_threshold_reported_at = (
        -1
    )  # track last filled count where threshold report was shown

    # Worker pool for anchor parallelization
    global_worker_pool = None

    # Set main process title
    if setproctitle:
        setproctitle(f"spw-solver: main [{puzzle_category} {puzzle_id}]")

    # Build slot indices before initializing workers (needed for worker initialization)
    build_slot_indices(initial_board)

    try:
        print("Using ANCHOR-DRIVEN PARALLEL solver...")
        sys.stdout.flush()

        # Solve with parallel anchors
        final_board = solve_with_parallel_anchors()

    except KeyboardInterrupt:
        print("Exiting...")
        if global_worker_pool:
            global_worker_pool.terminate()
            global_worker_pool.join()
        sys.exit()
    finally:
        # Cleanup worker pool
        if global_worker_pool:
            try:
                global_worker_pool.close()
                global_worker_pool.join()
            except Exception as e:
                print(f"Warning: Error during worker pool cleanup: {e}")
                try:
                    global_worker_pool.terminate()
                    global_worker_pool.join()
                except:
                    pass

    if final_board != []:
        print("\nSolution found!")
        print("Final board:")
        display_board(final_board)

        # Verify solution uses only letters from bag
        # Only check NEW letters (final_board minus initial_board)
        initial_letters = "".join([ch for ch in initial_board if ch.isalpha()])
        final_letters = "".join([ch for ch in final_board if ch.isalpha()])

        # Remove initial letters from final to get only new letters
        new_letters = final_letters
        for letter in initial_letters:
            if letter in new_letters:
                new_letters = new_letters.replace(letter, "", 1)

        temp_bag = letter_bag
        verification_passed = True
        for letter in new_letters:
            if letter in temp_bag:
                temp_bag = temp_bag.replace(letter, "", 1)
            else:
                print(f"ERROR: Letter '{letter}' used but not in bag!")
                verification_passed = False

        # Also remove initial board letters to show truly unused letters
        for letter in initial_letters:
            if letter in temp_bag:
                temp_bag = temp_bag.replace(letter, "", 1)

        if verification_passed:
            print(
                f"✓ Verification passed: All {len(new_letters)} new letters are from the bag"
            )
            print(f"  Initial board had: {len(initial_letters)} letters")
            print(f"  Added from bag: {len(new_letters)} letters")
            if temp_bag:
                print(f"  Unused letters from bag: {temp_bag}")
            else:
                print(f"  All bag letters used!")
        else:
            print("✗ Verification FAILED: Solution uses letters not in bag!")

        elapsed = time.time() - start_time
        boards_per_sec = boards_processed_count / elapsed if elapsed > 0 else 0
        print(f"Time taken: {format_elapsed_time(elapsed)}")
        print(f"Total boards processed: {boards_processed_count:,}")
        print(f"Average: {boards_per_sec:,.0f} boards/second")
    else:
        print("\nNo solution found")
        elapsed = time.time() - start_time
        boards_per_sec = boards_processed_count / elapsed if elapsed > 0 else 0
        print(f"Time taken: {format_elapsed_time(elapsed)}")
        print(f"Total boards processed: {boards_processed_count:,}")
        print(f"Average: {boards_per_sec:,.0f} boards/second")
