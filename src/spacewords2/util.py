"""Utility functions for Spacewords."""


def time_str(seconds: float) -> str:
    """Convert a time duration in seconds to a human-readable string.

    Args:
        seconds: Time duration in seconds.

    Returns:
        A string formatted as "HH:MM:SS.sss".
    """
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{secs:05.2f}"


def int_comma(n: int) -> str:
    """Format an integer with commas as thousands separators.

    Args:
        n: The integer to format.

    Returns:
        A string representation of the integer with commas.
    """
    return f"{n:,}"
