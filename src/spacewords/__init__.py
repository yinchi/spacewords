"""Spacewords Puzzle Solver.

Attempts to place a set of tiles onto a board so that all words formed are valid
according to a given dictionary.  The board may contain pre-placed tiles and/or
blocked spaces (denoted by '#').  Uses backtracking to explore possible placements.
"""

from sys import argv, exit

from .puzzle_config import load_configs
from .solver import solver


def main() -> None:
    """Main entry point for the Spacewords solver."""
    # Expect a single argument: path to the configuration file
    if len(argv) != 2:
        print("Usage: python -m spacewords <path_to_config_file>")
        exit(1)
    config_path = argv[1]
    configs = load_configs(config_path)

    for config in configs:
        solver.run(config)
