"""Central Orchestrator agent implementing the blackboard state-passing pattern."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from . import config
from .agents.api_agents import ArbeitnowAgent, RemoteOKAgent, RemotiveAgent
from .agents.platform_searcher import PlatformSearcher
from .agents.resume_analyzer import ResumeAnalyzer
from .agents.vetting import MatchVettingAgent
from .models import CandidateProfile, JobListing, JobSearchCriteria
from .writer import write_excel

log = logging.getLogger(__name__)


class ResumeJobOrchestrator:
    """Coordinates ingestion, parallel querying, dedupe, pre-filter, vetting and reporting."""

    def __init__(self, llm=None):
        if llm is None:
            from .llm import GroqLLM

            llm = GroqLLM()
        self.llm = llm
        self.blackboard = {"criteria": None, "profile": None, "jobs": [], "metrics": {}}
        self.analyzer = ResumeAnalyzer(self.llm)
        self.sources = [PlatformSearcher(), RemotiveAgent(), RemoteOKAgent(), ArbeitnowAgent()]
        self.vetter = MatchVettingAgent(self.llm)

    def run(
        self,
        resume_path: str,
        criteria: JobSearchCriteria = None,
        max_evals: int = config.MAX_EVALS_DEFAULT,
        output_path: str = "outputs/job_matches.xlsx",
    ) -> dict:
        started = time.time()
        criteria = criteria or JobSearchCriteria()
        self.blackboard["criteria"] = criteria

        # Phase 1 (Ingest): resume -> CandidateProfile on the blackboard.
        log.info("Phase 1: analyzing resume %s", resume_path)
        profile = self.analyzer.analyze(resume_path)
        self.blackboard["profile"] = profile
        queries = profile.search_queries or profile.job_titles or criteria.keywords
        log.info("Search queries: %s", queries)

        # Phase 2 (Querying): all source agents run in parallel.
        log.info("Phase 2: querying %d sources in parallel", len(self.sources))
        jobs: List[JobListing] = []
        with ThreadPoolExecutor(max_workers=len(self.sources)) as pool:
            futures = {
                pool.submit(source.search, queries, criteria.max_results_per_source): source
                for source in self.sources
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    found = future.result()
                    log.info("%s returned %d listings", source.name, len(found))
                    jobs.extend(found)
                except Exception as exc:
                    log.warning("Source %s failed: %s", source.name, exc)

        # Phase 3 (Filtering & Vetting): dedupe, pre-filter, then parallel LLM screens.
        jobs = self.deduplicate(jobs)
        candidates = self.prefilter(jobs, profile, criteria)[:max_evals]
        log.info("Phase 3: vetting %d of %d unique listings", len(candidates), len(jobs))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda job: self.vetter.evaluate(profile, job), candidates))
        candidates.sort(key=lambda job: job.fit_score, reverse=True)
        self.blackboard["jobs"] = candidates

        # Phase 4 (Reporting): styled Excel export.
        log.info("Phase 4: writing report")
        path = write_excel(profile, candidates, output_path)

        metrics = {
            "total_found": len(jobs),
            "evaluated": len(candidates),
            "strong_fits": sum(1 for j in candidates if j.fit_decision == "Strong Fit"),
            "elapsed_seconds": round(time.time() - started, 1),
            "output_path": path,
        }
        self.blackboard["metrics"] = metrics
        return {"profile": profile, "jobs": candidates, "metrics": metrics, "output_path": path}

    @staticmethod
    def deduplicate(jobs: List[JobListing]) -> List[JobListing]:
        """Drop listings that share the same URL (or title+company when URL is missing)."""
        seen, unique = set(), []
        for job in jobs:
            key = job.url.rstrip("/").lower() or (job.title.lower(), job.company.lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(job)
        return unique

    @staticmethod
    def prefilter(
        jobs: List[JobListing],
        profile: CandidateProfile,
        criteria: JobSearchCriteria,
    ) -> List[JobListing]:
        """Rank listings by keyword overlap to conserve the LLM API quota."""
        if criteria.remote_only:
            jobs = [j for j in jobs if "remote" in f"{j.location} {j.description}".lower()]

        tokens = set()
        for text in profile.job_titles + profile.skills + criteria.keywords:
            tokens.update(t.lower() for t in text.split() if len(t) > 2)

        def overlap(job: JobListing) -> int:
            blob = f"{job.title} {job.description}".lower()
            return sum(1 for t in tokens if t in blob)

        scored = [(overlap(job), job) for job in jobs]
        relevant = [(s, j) for s, j in scored if s > 0] or scored
        relevant.sort(key=lambda pair: pair[0], reverse=True)
        return [job for _, job in relevant]
