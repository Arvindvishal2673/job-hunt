import logging
from typing import List
import requests
from ..models import JobListing
from .base import JobSourceAgent
from .. import config

log = logging.getLogger(__name__)

# List of Greenhouse and Lever company tokens in India that are verified working
GREENHOUSE_COMPANIES = [
    ("groww", "Groww"),
    ("credave", "CredAvenue / Yubi"),
    ("postman", "Postman"),
    ("browserstack", "BrowserStack"),
    ("chargebee", "Chargebee"),
    ("delhivery", "Delhivery"),
    ("innovaccer", "Innovaccer"),
    ("unacademy", "Unacademy"),
]

LEVER_COMPANIES = [
    ("cred", "CRED"),
    ("pocketfm", "PocketFM"),
]


class DirectATSAgent(JobSourceAgent):
    name = "direct_ats"

    def search(self, queries: List[str], max_results: int = 25) -> List[JobListing]:
        listings: List[JobListing] = []
        
        # Helper to do cheap overlap check
        def _matches(text: str, search_queries: List[str]) -> bool:
            text = text.lower()
            for query in search_queries:
                tokens = [t for t in query.lower().split() if len(t) > 2]
                if tokens and any(t in text for t in tokens):
                    return True
            return not search_queries

        # 1. Query Greenhouse boards
        for company_id, company_name in GREENHOUSE_COMPANIES:
            url = f"https://boards-api.greenhouse.io/v1/boards/{company_id}/jobs?content=true"
            try:
                res = requests.get(url, timeout=config.REQUEST_TIMEOUT)
                if res.status_code == 200:
                    jobs = res.json().get("jobs", [])
                    for job in jobs:
                        title = job.get("title", "")
                        content = job.get("content", "")
                        # Quick match check
                        blob = f"{title} {content}"
                        if not _matches(blob, queries):
                            continue
                        
                        listings.append(
                            JobListing(
                                title=title,
                                company=company_name,
                                location=job.get("location", {}).get("name", "India"),
                                url=job.get("absolute_url", ""),
                                source="Greenhouse",
                                description=content[:2000],
                            )
                        )
            except Exception as exc:
                log.warning("Greenhouse direct fetch failed for %s: %s", company_name, exc)

        # 2. Query Lever boards
        for company_id, company_name in LEVER_COMPANIES:
            url = f"https://api.lever.co/v0/postings/{company_id}"
            try:
                res = requests.get(url, timeout=config.REQUEST_TIMEOUT)
                if res.status_code == 200:
                    jobs = res.json()
                    for job in jobs:
                        title = job.get("title", "")
                        desc = job.get("description", "")
                        lists = job.get("lists", [])
                        list_text = " ".join([item.get("content", "") for sublist in lists for item in sublist.get("items", [])])
                        blob = f"{title} {desc} {list_text}"
                        if not _matches(blob, queries):
                            continue
                        
                        loc_dict = job.get("categories", {})
                        location = loc_dict.get("location", "India")
                        
                        listings.append(
                            JobListing(
                                title=title,
                                company=company_name,
                                location=location,
                                url=job.get("applyUrl", ""),
                                source="Lever",
                                description=desc[:2000],
                            )
                        )
            except Exception as exc:
                log.warning("Lever direct fetch failed for %s: %s", company_name, exc)

        return listings[:max_results]
