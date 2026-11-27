"""RCA pipeline package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rca_pipeline.ranking import Candidate as Candidate
    from rca_pipeline.ranking import rank as rank


def __getattr__(name: str):
    if name in {"Candidate", "rank"}:
        from rca_pipeline import ranking

        return getattr(ranking, name)
    raise AttributeError(name)
