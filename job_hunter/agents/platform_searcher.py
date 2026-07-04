"""Multi-Platform Search agent: DuckDuckGo `site:` search across 15 job platforms."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from urllib.parse import urlparse

from ..models import JobListing
from .base import JobSourceAgent

log = logging.getLogger(__name__)

PLATFORM_DOMAINS = {
    "linkedin.com": "LinkedIn",
    "indeed.com": "Indeed",
    "ziprecruiter.com": "ZipRecruiter",
    "glassdoor.com": "Glassdoor",
    "dice.com": "Dice",
    "monster.com": "Monster",
    "careerbuilder.com": "CareerBuilder",
    "simplyhired.com": "SimplyHired",
    "remote.co": "Remote.co",
    "weworkremotely.com": "We Work Remotely",
    "remotive.com": "Remotive",
    "remoteok.com": "RemoteOK",
    "arbeitnow.com": "Arbeitnow",
    "reed.co.uk": "Reed",
    "adzuna.com": "Adzuna",
}


def domain_to_platform(url: str) -> str:
    """Map a result URL back to its origin platform name."""
    host = urlparse(url).netloc.lower()
    for domain, name in PLATFORM_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return "Web"


def _site_groups(group_count: int = 3) -> List[List[str]]:
    """Split the 15 platform domains into distinct `site:` search groups."""
    domains = list(PLATFORM_DOMAINS)
    size = -(-len(domains) // group_count)  # ceiling division
    return [domains[i : i + size] for i in range(0, len(domains), size)]


class PlatformSearcher(JobSourceAgent):
    """Runs grouped `site:` queries in parallel via DuckDuckGo Search."""

    name = "platform_searcher"

    def search(self, queries: List[str], max_results: int = 25) -> List[JobListing]:
        try:
            import duckduckgo_search  # noqa: F401
        except ImportError:
            log.warning(
                "duckduckgo-search is not installed; skipping platform search. "
                "Run `pip install duckduckgo-search` to enable it."
            )
            return []

        tasks = []
        for query in queries:
            for group in _site_groups():
                sites = " OR ".join(f"site:{d}" for d in group)
                tasks.append(f"{query} ({sites})")

        listings: List[JobListing] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(self._run_query, t, max_results): t for t in tasks}
            for future in as_completed(futures):
                try:
                    listings.extend(future.result())
                except Exception as exc:  # network failures must not crash the run
                    log.warning("Platform search failed for %r: %s", futures[future], exc)
        return listings

    @staticmethod
    def _run_query(query: str, max_results: int) -> List[JobListing]:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                url = hit.get("href", "")
                if not url:
                    continue
                results.append(
                    JobListing(
                        title=hit.get("title", "").strip(),
                        url=url,
                        source=domain_to_platform(url),
                        description=hit.get("body", ""),
                    )
                )
        return results
