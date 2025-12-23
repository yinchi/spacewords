"""Loader for configuration files."""

from collections import Counter
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np

from spacewords.solver.config import config as solver_config


@dataclass
class PuzzleConfig:
    """A puzzle configuration."""

    category: Literal["daily", "weekly", "monthly"]
    """The puzzle category."""

    puzzle_id: int
    """An integer representing the puzzle ID."""

    dims: tuple[int, int]
    """The height and width of the puzzle grid."""

    tiles: Counter[str]
    """A multiset of available tiles for filling the board."""

    board_str: str
    """The initial state of the puzzle board, in row-major order.

    Filled cells are represented by uppercase letters. Empty cells are
    represented by dots ('.').  Cells that are blocked (must remain empty)
    are represented by hash signs ('#').
    """

    def __post_init__(self) -> None:
        """Validate the board."""
        # Ensure the board shape matches the specified dimensions
        height, width = self.dims
        assert len(self.board_str) == height * width, (
            f"Board string length does not match dimensions {self.dims}."
        )

        # Ensure the board contains only valid characters (A-Z, ., #)
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ.#")
        board_chars = set(self.board_str)
        assert board_chars.issubset(valid_chars), (
            f"Board contains invalid characters: {board_chars - valid_chars}"
        )

        # Ensure there are no blocked cells ('#') in the anchor column (currenlty always column 0)
        for idx in range(solver_config.anchor_col, len(self.board_str), width):
            if self.board_str[idx] == "#":
                raise ValueError(
                    f"Blocked cells ('#') are not allowed in column {solver_config.anchor_col}."
                )

        # Ensure board is connected (blocked cells do not split the board)
        if not self.is_connected():
            raise ValueError("The board is not fully connected due to blocked cells ('#').")

    def __str__(self) -> str:
        """Return a string representation of the Config."""
        return (
            f"{self.category}/{self.puzzle_id} ({self.dims[0]}x{self.dims[1]}): "
            f"{''.join(sorted(self.tiles.elements()))}\n"
            f"{self.board_str}"
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Config for serialization.

        This is useful for supplying the Config to child processes via
        `multiprocessing`, which requires arguments to be pickleable.
        """
        return {
            "category": self.category,
            "puzzle_id": self.puzzle_id,
            "dims": self.dims,
            "tiles": dict(self.tiles),
            "board": self.board_str,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PuzzleConfig":
        """Create a Config instance from a dictionary representation."""
        height, width = data["dims"]
        board_str = data["board"]
        assert len(board_str) == height * width, "Board string length does not match dimensions."

        return cls(
            category=data["category"],
            puzzle_id=data["puzzle_id"],
            dims=data["dims"],
            tiles=Counter(data["tiles"]),
            board_str=board_str,
        )

    def is_connected(self) -> bool:
        """Check if the board is fully connected ("#" cells do not split the board)."""
        visited = np.zeros(self.dims, dtype=bool)

        # Convert board string to 2D array for easier indexing
        board = np.array(list(self.board_str)).reshape(self.dims)

        # Find the first non-blocked cell to start DFS
        def _find_start() -> tuple[int, int] | None:
            for r, c in np.ndindex(self.dims):
                if board[r, c] != "#":
                    return (r, c)
            return None

        start = _find_start()
        if start is None:
            return True  # Entire board is blocked

        # Depth-First Search (DFS) to mark all reachable cells
        stack = [start]
        while stack:
            r, c = stack.pop()
            if visited[r, c]:
                continue
            visited[r, c] = True

            # Check neighbors (up, down, left, right)
            for delta_r, delta_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r_new, c_new = r + delta_r, c + delta_c
                if 0 <= r_new < self.dims[0] and 0 <= c_new < self.dims[1]:
                    if board[r_new, c_new] != "#" and not visited[r_new, c_new]:
                        stack.append((r_new, c_new))

        # Check if all non-blocked cells were visited
        for r, c in np.ndindex(self.dims):
            if board[r, c] != "#" and not visited[r, c]:
                return False
        return True


def clean(board_str: str) -> str:
    """Clean the board string by removing whitespace and converting all letters to uppercase."""
    return board_str.replace(" ", "").replace("\n", "").upper()


def load_configs(configs_path: PathLike, *, print_boards: bool = False) -> list[PuzzleConfig]:
    """Load configuration file from the given path.

    Args:
        configs_path (PathLike): Path to the configuration file.
        print_boards (bool): Whether to print loaded boards for debugging.
    """
    configs = []

    path = Path(configs_path).resolve()
    print()
    print(f"Loading configs from {path}")
    print()
    category, puzzle_id = path.parts[-2:]  # Get last two parts: category, puzzle_id.txt
    puzzle_id = int(puzzle_id.removesuffix(".txt"))

    with open(path, "r", encoding="utf-8") as f:
        # Get dimensions from first line
        first_line = f.readline().strip()
        try:
            height, width = map(int, first_line.split())
        except ValueError:
            # Covers both incorrect number of values and non-integer values
            raise ValueError(f"Invalid dimensions line: '{first_line}'") from None

        # Read the tile list from the second line
        tile_line = f.readline().strip()
        tiles = Counter(tile_line.upper())

        # Skip first blank line
        assert f.readline() == "\n", "Expected a blank line after tile list."

        # Read the board lines, each board is separated by a blank line
        while True:
            board_lines = []
            while True:
                line = f.readline()
                if not line or line.strip() == "":
                    break
                board_lines.append(line.strip())

            if not board_lines:
                break  # No more boards to read

            board_str = clean("\n".join(board_lines))
            config = PuzzleConfig(
                category=category,
                puzzle_id=puzzle_id,
                dims=(height, width),
                tiles=tiles,
                board_str=board_str,
            )
            if print_boards:
                print(config)
                print(config.to_dict())
                print()
            configs.append(config)

    return configs
