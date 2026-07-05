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
    "boards.greenhouse.io": "Greenhouse",
    "jobs.lever.co": "Lever",
}

PLATFORM_DOMAINS_INDIA = {
    "in.linkedin.com": "LinkedIn India",
    "linkedin.com": "LinkedIn",
    "in.indeed.com": "Indeed India",
    "indeed.com": "Indeed",
    "naukri.com": "Naukri",
    "instahyre.com": "Instahyre",
    "internshala.com": "Internshala",
    "cuvette.tech": "Cuvette",
    "wellfound.com": "Wellfound",
    "glassdoor.co.in": "Glassdoor India",
    "foundit.in": "Foundit India",
    "boards.greenhouse.io": "Greenhouse",
    "jobs.lever.co": "Lever",
    "freshersworld.com": "Freshersworld",
    "shine.com": "Shine",
    "hirist.tech": "Hirist",
    "timesjobs.com": "TimesJobs",
}


class PlatformSearcher(JobSourceAgent):
    """Runs grouped `site:` queries in parallel via DuckDuckGo Search."""

    name = "platform_searcher"

    def __init__(self, target_india_only: bool = False):
        self.target_india_only = target_india_only

    def _get_domains(self) -> dict:
        return PLATFORM_DOMAINS_INDIA if self.target_india_only else PLATFORM_DOMAINS

    def _site_groups(self, group_count: int = 3) -> List[List[str]]:
        """Split the platform domains into distinct `site:` search groups."""
        domains = list(self._get_domains())
        size = -(-len(domains) // group_count)  # ceiling division
        return [domains[i : i + size] for i in range(0, len(domains), size)]

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
            tasks.append(f"{query} jobs")

        # Set up region and time limits dynamically
        region = "in-en" if self.target_india_only else "wt-wt"
        timelimit = "m" if self.target_india_only else None  # "m" gets jobs posted in the past month

        listings: List[JobListing] = []
        # Use lower worker count (max_workers=2) to serialize queries and prevent rate limits
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self._run_query, t, max_results, region, timelimit): t 
                for t in tasks
            }
            for future in as_completed(futures):
                try:
                    listings.extend(future.result())
                except Exception as exc:  # network failures must not crash the run
                    log.warning("Platform search failed for %r: %s", futures[future], exc)
        return listings

    def _run_query(self, query: str, max_results: int, region: str, timelimit: str) -> List[JobListing]:
        from duckduckgo_search import DDGS
        import time
        import random

        # Introduce random delay to prevent IP rate-limiting by the search engine
        time.sleep(random.uniform(1.5, 3.5))

        results = []
        with DDGS() as ddgs:
            for hit in ddgs.text(query, region=region, timelimit=timelimit, max_results=max_results):
                url = hit.get("href", "")
                if not url:
                    continue
                results.append(
                    JobListing(
                        title=hit.get("title", "").strip(),
                        url=url,
                        source=self.domain_to_platform(url),
                        description=hit.get("body", ""),
                    )
                )
        return results

    def domain_to_platform(self, url: str) -> str:
        """Map a result URL back to its origin platform name."""
        host = urlparse(url).netloc.lower()
        all_domains = {**PLATFORM_DOMAINS, **PLATFORM_DOMAINS_INDIA}
        for domain, name in all_domains.items():
            if host == domain or host.endswith("." + domain):
                return name
        return "Web"
