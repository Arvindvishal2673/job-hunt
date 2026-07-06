"""Central Orchestrator implementing the full agentic pipeline.

Pipeline phases:
  1.   Resume → CandidateProfile  (ResumeAnalyzer)
  1.5  Profile → Search Queries   (SearchStrategyAgent)
  2a.  Profile → Source Selection (PlannerAgent — LLM tool-calling)
  2.   Parallel job ingestion from LLM-selected sources
  2b.  ReAct loop: Observe results → Reason → Act (refine or done)
  3.   Dedupe → Pre-filter → Parallel LLM vetting (MatchVettingAgent)
  4.   Styled Excel export
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from . import config
from .agents.api_agents import AdzunaAgent, ArbeitnowAgent, RemoteOKAgent, RemotiveAgent
from .agents.apify_agent import ApifyLinkedInAgent
from .agents.ats_agent import DirectATSAgent
from .agents.planner import PlannerAgent
from .agents.reflector import ReflectionAgent, MAX_REACT_ITERATIONS
from .agents.resume_analyzer import ResumeAnalyzer
from .agents.search_strategy import SearchStrategyAgent
from .agents.vetting import MatchVettingAgent
from .models import CandidateProfile, JobListing, JobSearchCriteria
from .writer import write_excel

log = logging.getLogger(__name__)


class ResumeJobOrchestrator:
    """Coordinates all agents via the blackboard state-passing pattern."""

    def __init__(self, llm=None):
        if llm is None:
            from .llm import GroqLLM
            llm = GroqLLM()
        self.llm = llm
        self.blackboard = {"criteria": None, "profile": None, "jobs": [], "metrics": {}}
        self.analyzer = ResumeAnalyzer(self.llm)
        self.strategy_agent = SearchStrategyAgent(self.llm)
        self.planner = PlannerAgent(self.llm)
        self.reflector = ReflectionAgent(self.llm)
        self.vetter = MatchVettingAgent(self.llm)

    def _build_all_sources(self, target_india_only: bool) -> list:
        """Construct the full pool of source agents the planner can choose from."""
        if target_india_only:
            return [
                ApifyLinkedInAgent(target_india_only=True),
                AdzunaAgent(),
                DirectATSAgent(),
            ]
        return [
            ApifyLinkedInAgent(target_india_only=False),
            RemotiveAgent(),
            RemoteOKAgent(),
            ArbeitnowAgent(),
            AdzunaAgent(),
            DirectATSAgent(),
        ]

    def _run_sources_parallel(
        self,
        sources: list,
        queries: List[str],
        max_results: int,
    ) -> List[JobListing]:
        """Run a set of source agents in parallel and collect all results."""
        jobs: List[JobListing] = []
        with ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = {
                pool.submit(source.search, queries, max_results): source
                for source in sources
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    found = future.result()
                    log.info("%s returned %d listings", source.name, len(found))
                    jobs.extend(found)
                except Exception as exc:
                    log.warning("Source %s failed: %s", source.name, exc)
        return jobs

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

        # ── Phase 1: Resume → CandidateProfile ───────────────────────────────
        log.info("Phase 1: analyzing resume %s", resume_path)
        profile: CandidateProfile = self.analyzer.analyze(resume_path)

        # ── Phase 1.5: Profile → Search Queries ──────────────────────────────
        log.info("Phase 1.5: generating search strategy")
        profile.search_queries = self.strategy_agent.generate_queries(profile)
        self.blackboard["profile"] = profile
        queries = profile.search_queries or profile.job_titles or criteria.keywords
        log.info("Initial search queries: %s", queries)

        # ── Phase 2a: PlannerAgent selects which sources to activate ─────────
        log.info("Phase 2a: PlannerAgent selecting job sources via tool-calling...")
        all_sources = self._build_all_sources(criteria.target_india_only)
        active_sources = self.planner.select_sources(
            profile, all_sources, criteria.target_india_only
        )
        profile.activated_sources = [type(s).__name__ for s in active_sources]
        log.info(
            "Phase 2a complete. LLM activated %d/%d sources: %s",
            len(active_sources), len(all_sources), profile.activated_sources,
        )

        # ── Phase 2: Parallel ingestion with ReAct refinement loop ───────────
        log.info("Phase 2: querying %d sources in parallel", len(active_sources))
        jobs: List[JobListing] = self._run_sources_parallel(
            active_sources, queries, criteria.max_results_per_source
        )

        # ── Phase 2b: ReAct Observe → Reason → Act loop ──────────────────────
        for iteration in range(MAX_REACT_ITERATIONS):
            action, new_queries = self.reflector.reflect(
                profile, queries, jobs, iteration
            )
            profile.react_iterations = iteration + 1

            if action == "done":
                log.info("ReAct loop complete after %d iteration(s).", iteration + 1)
                break

            # action == "refine": update queries and search again
            log.info(
                "ReAct iteration %d: refining search with new queries: %s",
                iteration + 1, new_queries,
            )
            queries = new_queries
            profile.search_queries = new_queries
            new_jobs = self._run_sources_parallel(
                active_sources, queries, criteria.max_results_per_source
            )
            jobs.extend(new_jobs)
            log.info(
                "ReAct iteration %d: found %d additional listings (%d total so far)",
                iteration + 1, len(new_jobs), len(jobs),
            )
        else:
            log.info("ReAct loop reached max iterations (%d).", MAX_REACT_ITERATIONS)

        # ── Phase 3: Dedupe → Pre-filter → Parallel LLM vetting ─────────────
        jobs = self.deduplicate(jobs)
        candidates = self.prefilter(jobs, profile, criteria)[:max_evals]
        log.info("Phase 3: vetting %d of %d unique listings", len(candidates), len(jobs))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda job: self.vetter.evaluate(profile, job), candidates))
        candidates.sort(key=lambda job: job.fit_score, reverse=True)
        self.blackboard["jobs"] = candidates

        # ── Phase 4: Styled Excel export ─────────────────────────────────────
        log.info("Phase 4: writing report")
        try:
            path = write_excel(profile, candidates, output_path)
        except Exception as exc:
            log.warning("Could not write Excel report: %s. Continuing without saving.", exc)
            path = ""

        metrics = {
            "total_found": len(jobs),
            "evaluated": len(candidates),
            "strong_fits": sum(1 for j in candidates if j.fit_decision == "Strong Fit"),
            "elapsed_seconds": round(time.time() - started, 1),
            "output_path": path,
            "activated_sources": profile.activated_sources,
            "react_iterations": profile.react_iterations,
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
