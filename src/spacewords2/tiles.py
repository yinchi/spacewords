"""Module for tile-related classes and functions."""

from array import array

TileBag = array[int]
"""An array of 26 integers representing counts of tiles A-Z."""


def create_tile_bag(tiles: str) -> TileBag:
    """Create a tile bag from a string of tiles.

    Each character in the string represents a tile. The function returns an array
    of integers representing the count of each tile (A-Z).

    Args:
        tiles: A string representing the tiles.

    Returns:
        An array of 26 integers, where each index corresponds to a letter A-Z
        and the value at that index is the count of that letter in the input string.

    Raises:
        ValueError: If the input string contains invalid characters or is too long.
    """
    bag = array("B", [0] * 26)  # Array of 26 unsigned 1-byte integers for A-Z

    # For type safety, limit the length of the input string
    if len(tiles) >= 256:
        raise ValueError("Tile string too long; must be less than 256 characters.")

    for ch in tiles.upper():
        if "A" <= ch <= "Z":
            index = ord(ch) - ord("A")
            bag[index] += 1
        else:
            raise ValueError(f"Invalid tile character: {ch}")
    return bag
