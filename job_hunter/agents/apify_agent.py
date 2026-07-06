"""Apify LinkedIn Jobs Scraper agent using Apify API runs."""

import logging
import time
from typing import List

import requests

from .. import config
from ..models import JobListing
from .base import JobSourceAgent

log = logging.getLogger(__name__)


class ApifyLinkedInAgent(JobSourceAgent):
    """Triggers and polls a cloud-based LinkedIn Jobs Scraper on the Apify platform."""

    name = "apify_linkedin"

    def __init__(self, target_india_only: bool = False):
        self.target_india_only = target_india_only

    def search(self, queries: List[str], max_results: int = 25) -> List[JobListing]:
        token = config.APIFY_API_TOKEN
        if not token:
            log.warning("APIFY_API_TOKEN not set; skipping Apify LinkedIn search.")
            return []

        if not queries:
            return []

        # Start the Apify run
        actor_id = "curious_coder~linkedin-jobs-scraper"
        run_url = f"https://api.apify.com/v2/acts/{actor_id}/runs?token={token}"
        
        # Determine target location based on criteria
        location = "India" if self.target_india_only else "United States"

        import urllib.parse
        urls = []
        for query in queries:
            q_escaped = urllib.parse.quote(query)
            l_escaped = urllib.parse.quote(location)
            urls.append(f"https://www.linkedin.com/jobs/search/?keywords={q_escaped}&location={l_escaped}")

        payload = {
            "urls": urls,
            "scrapeCompany": False,
            "count": max_results,
            "splitByLocation": False
        }

        log.info("Triggering Apify LinkedIn Jobs Scraper (location: %s, urls: %s)...", location, urls)
        try:
            res = requests.post(run_url, json=payload, timeout=config.REQUEST_TIMEOUT)
            res.raise_for_status()
            run_data = res.json().get("data", {})
        except Exception as exc:
            log.warning("Failed to trigger Apify actor: %s", exc)
            return []

        run_id = run_data.get("id")
        dataset_id = run_data.get("defaultDatasetId")
        if not run_id or not dataset_id:
            log.warning("Invalid response from Apify run trigger (missing run_id or dataset_id).")
            return []

        # Poll the actor run status until completion
        status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={token}"
        max_poll_time = 180  # 3 minutes maximum timeout
        poll_interval = 5
        elapsed = 0

        log.info("Apify actor run %s started. Polling status...", run_id)
        while elapsed < max_poll_time:
            try:
                status_res = requests.get(status_url, timeout=config.REQUEST_TIMEOUT)
                status_res.raise_for_status()
                run_status = status_res.json().get("data", {})
            except Exception as exc:
                log.warning("Failed to poll Apify run status: %s. Retrying...", exc)
                time.sleep(poll_interval)
                elapsed += poll_interval
                continue

            status = run_status.get("status")
            log.debug("Apify run status: %s", status)

            if status == "SUCCEEDED":
                log.info("Apify run completed successfully.")
                break
            elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                log.warning("Apify actor run failed with status: %s", status)
                return []

            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            log.warning("Apify run timed out before completion after %d seconds.", max_poll_time)
            return []

        # Retrieve items from the dataset
        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={token}"
        log.info("Retrieving scraped jobs from Apify dataset %s...", dataset_id)
        try:
            dataset_res = requests.get(dataset_url, timeout=config.REQUEST_TIMEOUT)
            dataset_res.raise_for_status()
            items = dataset_res.json()
        except Exception as exc:
            log.warning("Failed to download Apify dataset items: %s", exc)
            return []

        if not isinstance(items, list):
            log.warning("Unexpected dataset items format from Apify: %s", type(items))
            return []

        # Map parsed JSON to JobListing objects defensively
        listings: List[JobListing] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = item.get("positionName") or item.get("title") or item.get("jobTitle") or ""
            company = item.get("companyName") or item.get("company") or ""
            location_val = item.get("location") or item.get("locationName") or ""
            url = item.get("jobUrl") or item.get("url") or ""
            desc = item.get("description") or item.get("descriptionText") or ""

            if not title or not url:
                continue  # Skip invalid or incomplete listings

            listings.append(
                JobListing(
                    title=str(title).strip(),
                    company=str(company).strip(),
                    location=str(location_val).strip() or (location),
                    url=str(url).strip(),
                    source="LinkedIn",
                    description=str(desc)[:2000].strip(),
                )
            )

        log.info("Apify agent parsed %d LinkedIn listings.", len(listings))
        return listings
