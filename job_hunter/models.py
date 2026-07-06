"""State-sharing contracts passed between agents on the blackboard."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class JobSearchCriteria:
    """User-supplied constraints for a job hunt run."""

    keywords: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    remote_only: bool = False
    min_salary: Optional[int] = None
    max_results_per_source: int = 25
    target_india_only: bool = False


@dataclass
class CandidateProfile:
    """Structured profile produced by the Resume Analyzer agent."""

    summary: str = ""
    skills: List[str] = field(default_factory=list)
    seniority: str = "Unknown"
    job_titles: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    raw_text: str = ""
    # Agentic metadata written by PlannerAgent and ReflectionAgent
    activated_sources: List[str] = field(default_factory=list)
    react_iterations: int = 0



@dataclass
class JobListing:
    """A normalized job posting, enriched in-place by the vetting agent."""

    title: str
    company: str = ""
    location: str = ""
    url: str = ""
    source: str = ""
    description: str = ""
    salary: str = ""
    fit_score: float = 0.0
    fit_decision: str = ""
    fit_reasons: List[str] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
