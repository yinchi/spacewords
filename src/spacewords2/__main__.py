"""Main entry point for Spacewords Puzzle Solver version 2."""

import sys

from spacewords2.board import test_solver

print("===========================")
print("Spacewords Puzzle Solver v2")
print("===========================", flush=True)

n_rows, n_cols, n_copies = map(int, sys.argv[1:4])
test_solver(n_rows, n_cols, n_copies)
