"""Solver state representation for Spacewords puzzles."""

from collections import Counter
from datetime import datetime
from time import time

from spacewords.puzzle_config import PuzzleConfig
from spacewords.solver.utils import TIMESTAMP_FMT, get_word_counter, is_playable


class TaskArgs:
    """Wrapper for task arguments for the solver.

    Pickleable, so that it can be used with multiprocessing (passed to worker processes).
    """

    def __init__(self, *, config: PuzzleConfig, words: set[str]) -> None:
        """Initialize the solver with the given configuration and word list.

        Args:
            config (PuzzleConfig): The configuration for the puzzle.
            words (set[str]): The set of valid words.
        """
        self.puzzle_config = config.to_dict()
        """dict representing the puzzle configuration."""

        self.words = {w for w in words if is_playable(get_word_counter(w), tiles=config.tiles)}
        """Set of valid words."""

        self.start_time = time()
        """Timestamp when the solver started, in seconds since the epoch."""

    def summary(self) -> dict[str, object]:
        """Return a dictionary-based summary of the task arguments.

        Note: heavy derived structures like word maps and letter frequency tables are
        intentionally not stored on TaskArgs (to reduce per-task pickling overhead).
        """
        ret: dict[str, object] = {
            "puzzle_config": self.puzzle_config,
            "words_count": len(self.words),
            "start_time": datetime.fromtimestamp(self.start_time)
            .astimezone()
            .strftime(TIMESTAMP_FMT),
        }
        ret["puzzle_config"]["tiles"] = "".join(
            sorted(Counter(ret["puzzle_config"]["tiles"]).elements())
        )
        return ret
