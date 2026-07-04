"""Structured API agents: Remotive, RemoteOK and Arbeitnow public JSON feeds."""

import logging
from typing import List

import requests

from .. import config
from ..models import JobListing
from .base import JobSourceAgent

log = logging.getLogger(__name__)


def _matches(text: str, queries: List[str]) -> bool:
    """Cheap keyword overlap between a job blob and the search queries."""
    text = text.lower()
    for query in queries:
        tokens = [t for t in query.lower().split() if len(t) > 2]
        if tokens and any(t in text for t in tokens):
            return True
    return not queries


class RemotiveAgent(JobSourceAgent):
    name = "remotive"
    URL = "https://remotive.com/api/remote-jobs"

    def search(self, queries: List[str], max_results: int = 25) -> List[JobListing]:
        listings: List[JobListing] = []
        try:
            for query in queries or [""]:
                data = requests.get(
                    self.URL, params={"search": query}, timeout=config.REQUEST_TIMEOUT
                ).json()
                for job in data.get("jobs", [])[:max_results]:
                    listings.append(
                        JobListing(
                            title=job.get("title", ""),
                            company=job.get("company_name", ""),
                            location=job.get("candidate_required_location", "Remote"),
                            url=job.get("url", ""),
                            source="Remotive",
                            description=(job.get("description") or "")[:2000],
                            salary=job.get("salary", ""),
                        )
                    )
        except Exception as exc:
            log.warning("Remotive fetch failed: %s", exc)
            return []
        return listings


class RemoteOKAgent(JobSourceAgent):
    name = "remoteok"
    URL = "https://remoteok.com/api"

    def search(self, queries: List[str], max_results: int = 25) -> List[JobListing]:
        try:
            data = requests.get(
                self.URL,
                headers={"User-Agent": "job-hunter-agent"},
                timeout=config.REQUEST_TIMEOUT,
            ).json()
        except Exception as exc:
            log.warning("RemoteOK fetch failed: %s", exc)
            return []

        listings: List[JobListing] = []
        for job in data:
            if not isinstance(job, dict) or not job.get("position"):
                continue  # first element is a legal notice, not a job
            blob = f"{job.get('position', '')} {' '.join(job.get('tags', []))}"
            if not _matches(blob, queries):
                continue
            salary = ""
            if job.get("salary_min"):
                salary = f"${job['salary_min']:,} - ${job.get('salary_max') or job['salary_min']:,}"
            listings.append(
                JobListing(
                    title=job.get("position", ""),
                    company=job.get("company", ""),
                    location=job.get("location", "Remote") or "Remote",
                    url=job.get("url", ""),
                    source="RemoteOK",
                    description=(job.get("description") or "")[:2000],
                    salary=salary,
                )
            )
            if len(listings) >= max_results:
                break
        return listings


class ArbeitnowAgent(JobSourceAgent):
    name = "arbeitnow"
    URL = "https://www.arbeitnow.com/api/job-board-api"

    def search(self, queries: List[str], max_results: int = 25) -> List[JobListing]:
        try:
            data = requests.get(self.URL, timeout=config.REQUEST_TIMEOUT).json()
        except Exception as exc:
            log.warning("Arbeitnow fetch failed: %s", exc)
            return []

        listings: List[JobListing] = []
        for job in data.get("data", []):
            blob = f"{job.get('title', '')} {' '.join(job.get('tags', []))}"
            if not _matches(blob, queries):
                continue
            listings.append(
                JobListing(
                    title=job.get("title", ""),
                    company=job.get("company_name", ""),
                    location=job.get("location", ""),
                    url=job.get("url", ""),
                    source="Arbeitnow",
                    description=(job.get("description") or "")[:2000],
                )
            )
            if len(listings) >= max_results:
                break
        return listings
