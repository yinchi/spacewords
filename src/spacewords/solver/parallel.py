"""Implementation of the parallel solver: task distribution and worker management."""

import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import Writer
from pprint import pprint
from typing import Literal, TypedDict

from spacewords.board import Board
from spacewords.puzzle_config import PuzzleConfig
from spacewords.solver.config import config as solver_config
from spacewords.solver.task_args import TaskArgs
from spacewords.solver.utils import get_word_counter, is_playable
from spacewords.solver.worker import worker_task


class WorkerTaskPayload(TypedDict):
    """Payload submitted to worker processes."""

    anchor_word: str
    """Anchor word to place in the anchor column."""
    puzzle_config: dict
    """Dict representation of a PuzzleConfig."""


def solve_with_parallel_anchors(
    executor: ProcessPoolExecutor,
    task_args: TaskArgs,
    logf: Writer,
    *,
    word_map: dict[tuple[int, int, str], set[str]] | None = None,
) -> Board | None:
    """Solve the puzzle using anchor-column-based parallelism.

    Args:
        executor (ProcessPoolExecutor): Executor for managing worker processes.
        task_args (TaskArgs): Arguments to pass to worker tasks, in addition to the anchor word.
        logf: File object to log the solving process.
        word_map (dict[tuple[int, int, str], set[str]] | None): Precomputed word map for candidate
            generation and scoring.

    Returns:
        A solved Board if a solution is found, else None.
    """
    print("Solver config:", file=logf, flush=True)
    pprint(solver_config.model_dump(), stream=logf, width=120)
    print("Solver initialized with:", file=logf, flush=True)
    pprint(task_args.summary(), stream=logf, width=120)
    print("", file=logf, flush=True)
    print("#" * 80, file=logf, flush=True)
    print("", file=logf, flush=True)

    puzzle_config = PuzzleConfig.from_dict(task_args.puzzle_config)
    init_board = Board(
        puzzle_config.board_str,
        puzzle_config.dims[0],
        puzzle_config.dims[1],
    )

    # Generate anchor candidates
    print("Generating anchor candidates...")  # stdout
    if word_map is None:
        raise ValueError("word_map is required (pass the precomputed map from solver.py).")
    anchor_candidates: list[str] = gen_anchor_candidates(
        initial_board=init_board,
        letter_bag=puzzle_config.tiles,
        words=task_args.words,
        word_map=word_map,
        out=logf,
    )

    if not anchor_candidates:
        print("No anchor candidates found.", file=logf, flush=True)
        return None

    print(
        f"Generated {len(anchor_candidates)} anchor candidates.",
        file=logf,
        flush=True,
    )
    print("", file=logf, flush=True)

    # Display top 10 anchor candidates
    print("Top anchor candidates:", file=logf, flush=True)
    for i, candidate in enumerate(anchor_candidates[:10], start=1):
        print(f"  {i:2d}. {candidate}", file=logf, flush=True)

    # Start worker tasks
    print(
        "Starting parallel solver...",
        file=logf,
        flush=True,
    )
    tasks: list[WorkerTaskPayload] = [
        {
            "anchor_word": candidate,
            "puzzle_config": task_args.puzzle_config,
        }
        for candidate in anchor_candidates
    ]

    # Send tasks to worker processes, read results as they complete, and return (early exit)
    # the first successful solution found

    # We don't care which candidate provides the solution, so just use a list instead of a map
    futures = [executor.submit(_worker_task, task) for task in tasks]
    for future in as_completed(futures):
        try:
            result = future.result()
            if result.status == "no_solution":
                # print(
                #     f"Worker for anchor word '{result.anchor_word}' reported no solution found.",
                #     flush=True,
                # )
                pass
            elif result.status == "error":
                print(
                    f"Worker for anchor word '{result.anchor_word}' encountered an error:",
                    flush=True,
                )
                print(result.err_msg, file=logf, flush=True)
            # Should catch all remaining cases
            elif result.status == "success" and result.result is not None:
                print("Terminating remaining workers...", file=logf, flush=True)
                executor.shutdown(wait=False, cancel_futures=True)
                return result.result
            else:
                print(
                    f"Unexpected result for anchor word '{result.anchor_word}'.",
                    flush=True,
                )
        except Exception as e:
            print(f"Error retrieving worker result: {str(e)}", flush=True)
            print(traceback.format_exc(), file=logf, flush=True)

    # All tasks completed, no solution found
    print("All anchor candidates processed, no solution found.", file=logf, flush=True)
    return None


def gen_anchor_candidates(
    initial_board: Board,
    letter_bag: Counter[str],
    words: set[str],
    word_map: dict[tuple[int, int, str], set[str]],
    out: Writer,
) -> list[str]:
    """Generate anchor candidates from the initial board and available letters.

    Currently, each solver run uses a predefined anchor row (row 0), and each worker task
    also starts with a filled anchor column (currently fixed to column 0) from the list of
    generated candidates.

    Args:
        initial_board (Board): The initial puzzle board.
        letter_bag (Counter[str]): Available letters to use for anchors.
        words (set[str]): Set of valid words.
        word_map (dict): Mapping of `(word_length, position_in_word, character)` to sets of words.
        out (Writer): Output stream for logging.

    Returns:
        A list of anchor candidate strings.
    """
    # Extract the anchor column from the initial board
    anchor_col = solver_config.anchor_col  # Currently always 0
    col_chars = [initial_board[r, anchor_col] for r in range(initial_board.n_rows)]
    col_str = "".join(col_chars)
    print(f"Column {anchor_col}: {col_str}", file=out)

    # Find words that can fit in the anchor column
    words_by_length: dict[int, set[str]] = {}
    max_len = max(initial_board.n_rows, initial_board.n_cols)
    for w in words:
        lw = len(w)
        if lw <= max_len:
            words_by_length.setdefault(lw, set()).add(w)

    candidates = find_candidates(col_str, words, word_map, words_by_length=words_by_length)
    print(f"Found {len(candidates)} candidates matching {col_str}.", file=out)

    # Precompute per-row scoring specs for fast anchor scoring.
    # Each spec corresponds to the across-segment intersecting the anchor column in that row.
    empty: set[str] = set()

    def intersection_size(a: set[str], b: set[str]) -> int:
        if not a or not b:
            return 0
        if len(a) > len(b):
            a, b = b, a
        return sum(1 for x in a if x in b)

    # (row_idx, seg_len, anchor_pos_in_seg, weight, base_set)
    anchor_score_specs: list[tuple[int, int, int, float, set[str]]] = []
    # For rows where the across-segment would become fully filled after placing the anchor
    # letter (i.e., the anchor cell is the only blank), we can reject anchors that complete an
    # invalid across word.
    # (row_idx, prefix, suffix)
    complete_after_anchor_specs: list[tuple[int, str, str]] = []

    for row in range(initial_board.n_rows):
        left = anchor_col
        while left - 1 >= 0 and initial_board[row, left - 1] != "#":
            left -= 1
        right = anchor_col
        while right + 1 < initial_board.n_cols and initial_board[row, right + 1] != "#":
            right += 1

        seg_len = right - left + 1
        if seg_len <= 1:
            continue

        anchor_pos = anchor_col - left
        seg_chars = [initial_board[row, c] for c in range(left, right + 1)]

        # If the across segment is already fully filled, it doesn't contribute to scoring.
        if all(ch.isalpha() for ch in seg_chars):
            continue

        fixed_keys: list[tuple[int, int, str]] = []
        blanks = 0
        for pos, ch in enumerate(seg_chars):
            if ch == ".":
                blanks += 1
                continue
            if ch == "#":
                continue
            # Skip the anchor cell itself: it will be set by the anchor word.
            if pos == anchor_pos:
                continue
            fixed_keys.append((seg_len, pos, ch))

        length_words = words_by_length.get(seg_len, empty)
        base_sets: list[set[str]] = [length_words]
        base_sets.extend(word_map.get(k, empty) for k in fixed_keys)
        base_sets.sort(key=len)
        base_set = set(base_sets[0])
        for s in base_sets[1:]:
            base_set.intersection_update(s)
            if not base_set:
                break

        weight = 1.0 / float(2 ** (seg_len - 1))
        anchor_score_specs.append((row, seg_len, anchor_pos, weight, base_set))

        # If the segment has exactly one blank and it's at the anchor cell, then placing the
        # anchor letter completes the across word; reject anchors that create a non-word.
        if blanks == 1 and seg_chars[anchor_pos] == "." and all(
            (c.isalpha() or i == anchor_pos) for i, c in enumerate(seg_chars)
        ):
            prefix = "".join(seg_chars[:anchor_pos])
            suffix = "".join(seg_chars[anchor_pos + 1 :])
            complete_after_anchor_specs.append((row, prefix, suffix))

    # Filter candidates based on available letters
    valid_candidates = [w for w in candidates if is_playable(get_word_counter(w), tiles=letter_bag)]

    if complete_after_anchor_specs:
        valid_candidates = [
            w
            for w in valid_candidates
            if all(
                (prefix + w[row] + suffix) in words
                for row, prefix, suffix in complete_after_anchor_specs
            )
        ]
    print(
        f"{len(valid_candidates)} candidates can be formed with available letters.",
        file=out,
    )

    if solver_config.sort_anchors_by_score:
        def score_anchor(w: str) -> float:
            score = 0.0
            for row, seg_len, anchor_pos, weight, base_set in anchor_score_specs:
                letter_set = word_map.get((seg_len, anchor_pos, w[row]), empty)
                score += intersection_size(base_set, letter_set) * weight
            return score

        # Sort candidates by anchor score (higher is better)
        valid_candidates.sort(
            key=score_anchor,
            reverse=True,
        )

    if solver_config.max_anchor_candidates is not None:
        valid_candidates = valid_candidates[: solver_config.max_anchor_candidates]

    return valid_candidates


def find_candidates(
    pattern: str,
    words: set[str],
    word_map: dict[tuple[int, int, str], set[str]],
    *,
    words_by_length: dict[int, set[str]] | None = None,
) -> set[str]:
    """Find candidate words that can fit in the given pattern string.

    Args:
        pattern (str): The pattern string with fixed letters and placeholders.
            Example: "A..C.E" where '.' represents empty cells.
        words (set[str]): Set of valid words.
        word_map (dict): Mapping of `(word_length, position_in_word, character)` to sets of words.
        words_by_length (dict[int, set[str]] | None): Optional mapping of word length to the set
            of words of that length. When provided, avoids scanning the entire word list on each
            call.

    Returns:
        A set of candidate words.
    """
    length = len(pattern)
    empty: set[str] = set()
    if words_by_length is not None:
        length_words = words_by_length.get(length, empty)
    else:
        length_words = {w for w in words if len(w) == length}

    constraint_sets: list[set[str]] = [length_words]
    for pos, ch in enumerate(pattern):
        if ch.isalpha():
            constraint_sets.append(word_map.get((length, pos, ch), empty))

    # Intersect starting from the smallest set to reduce work.
    constraint_sets.sort(key=len)
    candidates = set(constraint_sets[0])
    for s in constraint_sets[1:]:
        candidates.intersection_update(s)
        if not candidates:
            break
    return candidates


@dataclass
class Result:
    """Wrapper for worker task results."""

    anchor_word: str
    status: Literal["success", "no_solution", "error"]
    result: Board | None
    err_msg: str | None = None


def _worker_task(args: WorkerTaskPayload) -> Result:
    """Worker task to solve the puzzle with a given anchor word.

    Args:
        args (dict): Dictionary received from `executor.submit` containing:
            - "anchor_word": The anchor word to place.
            - "puzzle_config": Dict representation of a PuzzleConfig.

    Returns:
        A Result wrapper.
    """
    try:
        anchor_word = args["anchor_word"]
        ret = worker_task(**args)
        return Result(
            anchor_word=anchor_word,
            status="success" if ret is not None else "no_solution",
            result=ret,
        )
    except Exception as e:
        return Result(
            anchor_word=args.get("anchor_word", "<missing>"),
            status="error",
            result=None,
            err_msg=f"Worker encountered an error: {str(e)}\n{traceback.format_exc()}",
        )
