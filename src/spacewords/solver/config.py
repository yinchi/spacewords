"""Spacewords solver configuration."""

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the environment file path, or None if not found
ENV_FILE = find_dotenv() or None


class SolverConfig(BaseSettings):
    """Configuration settings for the Spacewords solver."""

    deterministic: bool = True
    """Whether to run the solver in deterministic mode (a bit slower on average). Default: True."""

    max_workers: int | None = None
    """Maximum number of worker processes to use. If None (default), uses os.cpu_count()."""

    show_progress_threshold: float = 0.5
    """Fraction of board filled at which to start showing progress updates. Default: 0.5."""

    almost_solved_treshold: float = 0.9
    """Show a special notification when this fraction of the board is filled. Default: 0.9."""

    report_interval: int = 1000
    """Interval (in number of candidate boards) at which to report progress. Default: 1000."""

    use_anchor_parallelism: bool = True
    """Whether to use the anchor-column-based strategy for parallelism. Default: True."""

    max_anchor_candidates: int | None = None
    """Maximum number of anchor candidates to consider. If None (default), no limit."""

    max_con_prop_iters: int = 100
    """Maximum number of iterations for the constraint propagation step. Default: 100."""

    anchor_col: int = 8
    """Which column to use as the anchor (Default: 8).
    """

    sort_anchors_by_score: bool = True
    """Whether to sort anchor candidates by score before processing. Default is True."""

    prefer_rare_letters_in_slot_order: bool = True
    """Whether to prefer slots with rare letters when determining slot fill order.

    Default is True.
    """

    word_list_path: str = "spw-dict.txt"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="forbid",
    )


config = SolverConfig()
