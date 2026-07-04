"""Abstract interface every job source agent must implement."""

from abc import ABC, abstractmethod
from typing import List

from ..models import JobListing


class JobSourceAgent(ABC):
    """Structured interface invariance: all searchers share this contract."""

    name: str = "base"

    @abstractmethod
    def search(self, queries: List[str], max_results: int = 25) -> List[JobListing]:
        """Return normalized JobListing objects for the given queries.

        Implementations must never raise on network/dependency failures;
        they log the issue and return an empty list instead.
        """
