"""Offline test suite: mocks all LLM, file and network dependencies."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from job_hunter.agents.platform_searcher import (
    PLATFORM_DOMAINS,
    PlatformSearcher,
)
from job_hunter.agents.resume_analyzer import ResumeAnalyzer
from job_hunter.agents.search_strategy import SearchStrategyAgent
from job_hunter.agents.vetting import MatchVettingAgent
from job_hunter.llm import extract_json
from job_hunter.models import CandidateProfile, JobListing, JobSearchCriteria
from job_hunter.orchestrator import ResumeJobOrchestrator
from job_hunter.writer import write_excel


def make_profile() -> CandidateProfile:
    return CandidateProfile(
        summary="Backend engineer with API experience.",
        skills=["Python", "Django"],
        seniority="Senior",
        job_titles=["Senior Python Developer"],
        search_queries=["senior python developer remote"],
    )


class TestPlatformMapping:
    def test_known_domains(self):
        searcher = PlatformSearcher()
        assert searcher.domain_to_platform("https://www.linkedin.com/jobs/view/1") == "LinkedIn"
        assert searcher.domain_to_platform("https://remoteok.com/remote-jobs/1") == "RemoteOK"
        assert searcher.domain_to_platform("https://uk.indeed.com/viewjob?jk=1") == "Indeed"

    def test_unknown_domain_falls_back(self):
        searcher = PlatformSearcher()
        assert searcher.domain_to_platform("https://example.org/job") == "Web"

    def test_site_groups_cover_all_15_platforms(self):
        searcher = PlatformSearcher()
        groups = searcher._site_groups()
        flattened = [domain for group in groups for domain in group]
        assert sorted(flattened) == sorted(PLATFORM_DOMAINS)
        assert len(groups) == 3


class TestResumeAnalyzer:
    def test_reads_text_resume(self, tmp_path):
        resume = tmp_path / "resume.txt"
        resume.write_text("Python developer with 5 years experience.")
        analyzer = ResumeAnalyzer(llm=MagicMock())
        assert "Python developer" in analyzer.extract_text(resume)

    def test_unsupported_format_raises(self, tmp_path):
        resume = tmp_path / "resume.docx"
        resume.write_text("hi")
        with pytest.raises(ValueError):
            ResumeAnalyzer(llm=MagicMock()).extract_text(resume)

    def test_analyze_builds_profile_from_llm_json(self, tmp_path):
        resume = tmp_path / "resume.md"
        resume.write_text("# Jane Doe\nPython, Django, AWS")
        llm = MagicMock()
        llm.chat.return_value = json.dumps(
            {
                "summary": "Experienced Python engineer.",
                "skills": ["Python", "Django", "AWS"],
                "seniority": "Senior",
                "job_titles": ["Python Developer"],
            }
        )
        profile = ResumeAnalyzer(llm=llm).analyze(resume)
        assert profile.seniority == "Senior"
        assert not profile.search_queries  # empty by default, populated later
        assert "Django" in profile.skills


class TestSearchStrategyAgent:
    def test_generate_queries_builds_queries_from_llm_json(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps(
            {
                "search_queries": [
                    "python developer",
                    "django engineer",
                    "aws backend",
                    "extra ignored",
                ]
            }
        )
        profile = CandidateProfile(
            summary="Experienced Python engineer.",
            skills=["Python", "Django", "AWS"],
            seniority="Senior",
            job_titles=["Python Developer"],
        )
        queries = SearchStrategyAgent(llm=llm).generate_queries(profile)
        assert len(queries) == 3  # capped at 3 queries
        assert queries[0] == "python developer"
        assert queries[2] == "aws backend"


class TestVettingAgent:
    def test_successful_evaluation_updates_listing(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps(
            {
                "fit_score": 88,
                "fit_decision": "Strong Fit",
                "fit_reasons": ["Python and Django match"],
                "gaps_identified": ["No Kubernetes experience"],
            }
        )
        listing = JobListing(title="Python Dev")
        MatchVettingAgent(llm).evaluate(make_profile(), listing)
        assert listing.fit_score == 88.0
        assert listing.fit_decision == "Strong Fit"
        assert listing.gaps_identified == ["No Kubernetes experience"]

    def test_falls_back_on_invalid_json(self):
        llm = MagicMock()
        llm.chat.return_value = "sorry, I cannot respond in JSON today"
        listing = JobListing(title="Python Dev")
        MatchVettingAgent(llm).evaluate(make_profile(), listing)
        assert listing.fit_score == 0.0
        assert listing.fit_decision == "Decent Fit"

    def test_invalid_decision_normalized(self):
        llm = MagicMock()
        llm.chat.return_value = json.dumps({"fit_score": 50, "fit_decision": "Maybe??"})
        listing = JobListing(title="Python Dev")
        MatchVettingAgent(llm).evaluate(make_profile(), listing)
        assert listing.fit_decision == "Decent Fit"


class TestOrchestratorLogic:
    def test_deduplicates_by_normalized_url(self):
        jobs = [
            JobListing(title="A", url="https://x.com/j/1"),
            JobListing(title="B", url="https://x.com/j/1/"),
            JobListing(title="C", url="https://x.com/j/2"),
        ]
        assert len(ResumeJobOrchestrator.deduplicate(jobs)) == 2

    def test_prefilter_ranks_relevant_jobs_first(self):
        jobs = [
            JobListing(title="Bakery Assistant", description="bread and pastries"),
            JobListing(title="Senior Python Developer", description="Django REST APIs"),
        ]
        ranked = ResumeJobOrchestrator.prefilter(jobs, make_profile(), JobSearchCriteria())
        assert ranked[0].title == "Senior Python Developer"

    def test_prefilter_remote_only_filters_offices(self):
        jobs = [
            JobListing(title="Python Developer", location="Berlin office"),
            JobListing(title="Python Developer", location="Remote"),
        ]
        ranked = ResumeJobOrchestrator.prefilter(
            jobs, make_profile(), JobSearchCriteria(remote_only=True)
        )
        assert ranked
        assert all("remote" in job.location.lower() for job in ranked)


class TestUtilities:
    def test_extract_json_from_noisy_llm_response(self):
        assert extract_json('Here you go:\n{"a": 1}\nThanks!')["a"] == 1

    def test_extract_json_raises_without_object(self):
        with pytest.raises(ValueError):
            extract_json("no json here")

    def test_write_excel_creates_styled_file(self, tmp_path):
        job = JobListing(
            title="Python Dev",
            company="Acme",
            url="https://x.com/1",
            fit_score=90,
            fit_decision="Strong Fit",
            fit_reasons=["Great skill match"],
            gaps_identified=["None significant"],
        )
        out = write_excel(make_profile(), [job], str(tmp_path / "out.xlsx"))
        assert Path(out).exists()
