"""Classes and functions for representing the game board."""

from array import array
from typing import Iterable


class Board:
    """Store a 2D matrix of characters as a 1D list.

    Contains support for both 1D and 2D indexing.
    """

    def __init__(self, data: str | Iterable[str], rows: int, cols: int) -> None:
        data = list(data) if isinstance(data, str) else data
        self.data = array("w", data)
        self.n_rows = rows
        self.n_cols = cols

    def copy(self) -> "Board":
        """Generate a copy of the board."""
        return Board(self.data.__copy__(), self.n_rows, self.n_cols)

    def __str__(self) -> str:
        """Returns a string representation of the board."""
        return self.data.tounicode()

    def print(self, two_d: bool = True) -> None:
        """Print the board to the console."""
        if two_d:
            for row in range(self.n_rows):
                print("".join(self[row, col] for col in range(self.n_cols)))
        else:
            print(self.data.tounicode())

    def __getitem__(self, idx: int | tuple[int, int]) -> str:
        """Get cell content by 1D (row-major order) or 2D index."""
        if isinstance(idx, int):
            return self.data[idx]
        if isinstance(idx, tuple) and len(idx) == 2:
            row, col = idx
            return self.data[row * self.n_cols + col]
        raise IndexError("Invalid index type for Board.")

    def __setitem__(self, idx: int | tuple[int, int], value: str) -> None:
        """Set cell content by 1D (row-major order) or 2D index."""
        if isinstance(idx, int):
            self.data[idx] = value
            return
        if isinstance(idx, tuple) and len(idx) == 2:
            row, col = idx
            self.data[row * self.n_cols + col] = value
            return
        raise IndexError("Invalid index type for Board.")

    def get_2d_idx(self, one_d_idx: int) -> tuple[int, int]:
        """Convert a 1D index to a (row, col) tuple."""
        return divmod(one_d_idx, self.n_cols)

    def get_1d_idx(self, row: int, col: int) -> int:
        """Convert a (row, col) tuple to a 1D index."""
        return row * self.n_cols + col

    def across_range(self, start_idx: int, length: int) -> range:
        """Get the range of 1D indices for an across word starting at start_idx."""
        return range(start_idx, start_idx + length)

    def down_range(self, start_idx: int, length: int) -> range:
        """Get the range of 1D indices for a down word starting at start_idx."""
        return range(start_idx, start_idx + length * self.n_cols, self.n_cols)
