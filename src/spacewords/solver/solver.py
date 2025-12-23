"""Main solver module for Spacewords puzzles."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from io import Writer
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from time import time
from typing import TypeAlias

from spacewords.board import Board
from spacewords.puzzle_config import PuzzleConfig
from spacewords.solver.config import config as solver_config
from spacewords.solver.parallel import solve_with_parallel_anchors
from spacewords.solver.task_args import TaskArgs
from spacewords.solver.utils import TIMESTAMP_FMT
from spacewords.solver.worker import init_worker_globals
from spacewords.wordlist import create_word_map, get_letter_frequency, load_word_list

WordMap: TypeAlias = dict[tuple[int, int, str], set[str]]
LetterFrequency: TypeAlias = dict[str, float]


def get_executor(
    *,
    n_workers: int | None = None,
    board_str: str,
    initial_word_map: WordMap,
    initial_letter_frequency: LetterFrequency,
    words: set[str],
) -> ProcessPoolExecutor:
    """Get a ProcessPoolExecutor.

    Args:
        n_workers (int | None): Number of worker processes to create.  If None,
            defaults to number of CPU cores minus one.
        board_str (str): The board string, used to calculate total fillable cells.
        initial_word_map (WordMap): Initial word map to pass to workers.
        initial_letter_frequency (LetterFrequency): Initial letter frequency to pass to workers.
        words (set[str]): Filtered word list to pass to workers.

    Returns:
        A ProcessPoolExecutor instance for worker processes.
    """
    worker_ctr: Synchronized[int] = Value("i", 0)

    cpus = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() is None
    if n_workers is None:
        n_workers = max(1, cpus - 1)  # Leave one core free
    if n_workers > cpus:
        raise ValueError(
            f"Requested number of workers ({n_workers}) exceeds CPU count ({cpus})",
        )
    total_fillable_cells = sum(1 for ch in board_str if ch != "#")
    return ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker_globals,
        initargs=(
            worker_ctr,
            total_fillable_cells,
            time(),
            initial_word_map,
            initial_letter_frequency,
            words,
        ),
    )


def run(config: PuzzleConfig) -> None:
    """Run the solver on the given configuration.

    Args:
        config (Config): The configuration for the puzzle to solve.
    """
    print(f"config: {config}")

    # Get the list of blocking character indices
    block_idxs = [i for i, ch in enumerate(config.board_str) if ch == "#"]
    block_str = ",".join(map(str, block_idxs))
    # print(f"Blocking indices: {block_str}")

    # Get the top row, this serves an initial anchor for the solver
    board_width = config.dims[1]
    top_row_str = config.board_str[0:board_width].replace(".", "_").replace("#", "b")
    # print(f"Top row: {top_row_str}")

    logfile = Path(
        f"logs/{config.category}/{config.puzzle_id}-{config.dims[0]}x{config.dims[1]}"
        f"/{top_row_str}-{block_str}.log"
    )
    print(f"Log file: {logfile}")

    logfile.parent.mkdir(parents=True, exist_ok=True)

    with open(logfile, "w", encoding="utf-8") as logf:
        try:
            solve_one(config, logf=logf)
        except KeyboardInterrupt:
            print("Solver interrupted by user.", file=logf, flush=True)
            print("Solver interrupted by user.")
            sys.exit(1)
    print()


def solve_one(puzzle_config: PuzzleConfig, *, logf: Writer[str]) -> None:
    """Attempt to solve a Spacewords puzzle given a starting configuration.

    Args:
        puzzle_config (PuzzleConfig): The configuration for the puzzle to solve.
        logf: File object to log the solving process.
    """
    print(
        f"Selected puzzle: {puzzle_config.category}/{puzzle_config.puzzle_id}",
        file=logf,
        flush=True,
    )
    print(f"Dimensions: {puzzle_config.dims}", file=logf, flush=True)
    print(
        f"Tile bag: {''.join(sorted(puzzle_config.tiles.elements()))}",
        file=logf,
        flush=True,
    )
    print("Initial grid:", file=logf, flush=True)
    print("", file=logf, flush=True)
    for row_start in range(0, len(puzzle_config.board_str), puzzle_config.dims[1]):
        print(
            puzzle_config.board_str[row_start : row_start + puzzle_config.dims[1]],
            file=logf,
            flush=True,
        )
    print("", file=logf, flush=True)
    print(f"Available tiles: {puzzle_config.tiles.total()}", file=logf, flush=True)
    print(
        f"Number of blocked cells: {puzzle_config.board_str.count('#')}",
        file=logf,
        flush=True,
    )

    # Solver initialization object
    max_word_length = max(puzzle_config.dims)
    solver_init_args = TaskArgs(
        config=puzzle_config,
        words=load_word_list(max_len=max_word_length),
    )

    initial_word_map = create_word_map(solver_init_args.words)
    letter_frequency = get_letter_frequency(solver_init_args.words)

    # Start time as formatted string (in local timezone)
    start_time_str = (
        datetime.fromtimestamp(solver_init_args.start_time).astimezone().strftime(TIMESTAMP_FMT)
    )
    print(f"Start time: {start_time_str}", file=logf, flush=True)

    final_board: Board | None = None

    # Create a new executor for the solver
    with get_executor(
        n_workers=solver_config.max_workers,
        board_str=puzzle_config.board_str,
        initial_word_map=initial_word_map,
        initial_letter_frequency=letter_frequency,
        words=solver_init_args.words,
    ) as executor:
        try:
            print("Using anchor-driven parallel solver...", file=logf, flush=True)
            # Note: if solution found, `solve_with_parallel_anchors` will call
            # `executor.shutdown(wait=False)` for early exit of all worker processes.
            final_board = solve_with_parallel_anchors(
                executor,
                solver_init_args,
                logf,
                word_map=initial_word_map,
            )
        except KeyboardInterrupt as e:
            executor.shutdown(wait=False, cancel_futures=True)
            raise e
        except Exception as e:
            executor.shutdown(wait=False, cancel_futures=True)
            raise e

    if final_board:
        print("Solution found!", file=logf, flush=True)
        # TODO: validate and show solution, print stats, etc.
    else:
        print("No solution found.", file=logf, flush=True)
        # TODO: print stats, etc.
