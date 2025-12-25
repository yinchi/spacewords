"""Main entry point for Spacewords Puzzle Solver version 2."""

from time import time

from spacewords2.board import Board, ValueError
from spacewords2.slot import Direction, SlotPosition
from spacewords2.tiles import create_tile_bag
from spacewords2.util import int_comma, time_str

NROWS = 8
NCOLS = 9
LAYOUT = """
     F L A P J A C K S
 . . . . # . . # .
 . . # . . . . . .
 . . . . . # . . .
 . # . . . . . . .
 # . . . # . . . .
 . . . . . . # . .
 . . . # . . . . .
"""

TILES = "AAAAABBBCCDDEEEEEEEFFFFGGGHHIIJKLLMMMNNNOOOOPPPRRSSSTTTUUUUVWYY"

# Strip all whitespace from layout
LAYOUT = "".join(LAYOUT.split())

print(f"Layout after stripping whitespace: {LAYOUT}")

board = Board(layout=LAYOUT, n_rows=NROWS, n_cols=NCOLS)
tile_bag = create_tile_bag(TILES)

# Remove placed letters from tile bag
for ch in board.layout:
    if ord("A") <= ch <= ord("Z"):
        index = ch - ord("A")
        if tile_bag[index] == 0:
            raise ValueError(f"Not enough tiles to cover placed letter '{ch}' on the board.")
        tile_bag[index] -= 1

# Print tile bag after removing placed letters
tile_str = ""
for i in range(26):
    tile_str += chr(ord("A") + i) * tile_bag[i]

print("Initial board:")
board.print()
print(f"Initial tile bag (after removing placed letters): {tile_str}")

n_slots = len(board.slot_map)
print(f"Number of slots identified: {n_slots}")

try:
    solved_board = board.solve(tile_bag, SlotPosition(Direction.DOWN, 0, 8))
    print("Solved board:")
    solved_board.print()
    print(f"Boards checked: {int_comma(solved_board.solve_stats.boards_checked)}")
    print(f"Time taken: {time_str(time() - solved_board.solve_stats.start_time)}")
except ValueError as e:
    print("No solution found.")
    print(f"Reason: {e}")
    print(f"Boards checked: {int_comma(board.solve_stats.boards_checked)}")
    print(f"Time taken: {time_str(time() - board.solve_stats.start_time)}")
    print("Final board state:")
    board.print()
    print("Tile bag state:")
    tile_str = ""
    for i in range(26):
        tile_str += chr(ord("A") + i) * tile_bag[i]
    print(tile_str)
