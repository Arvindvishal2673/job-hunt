"""Search Strategy agent: takes a CandidateProfile and builds optimized search queries."""

import logging

from ..llm import extract_json
from ..models import CandidateProfile

log = logging.getLogger(__name__)

STRATEGY_SYSTEM = (
    "You are an expert technical recruiter and job search strategist. You analyze "
    "a candidate's profile and generate optimized job search queries. "
    "Respond with only a valid JSON object, no prose."
)

STRATEGY_PROMPT = """Given the candidate profile below, generate exactly 3 optimized search query strings for job boards.

Format the response as a JSON object with a single key "search_queries" mapping to a list of 3 strings.

Guidelines for generating queries:
1. Leverage the candidate's core skills, tools, and preferred target titles.
2. Use simple, direct, space-separated keyword phrases (e.g., "AI Developer Python PyTorch" or "Backend Developer Django FastAPI"). Do NOT use complex nested Boolean logic (AND/OR/NOT) or multiple parentheses, as search engines will fail or return zero results.
3. Tailor queries to target different aspects of the candidate's skillset (e.g., one general tech role, one framework/tool-specific query, one specialty query).
4. Keep seniority in mind (e.g. if seniority is "Junior", incorporate "junior" or "intern" terms into the query variations).

Candidate Profile:
- Summary: {summary}
- Skills: {skills}
- Seniority: {seniority}
- Job Titles: {job_titles}

Return ONLY the JSON object.
"""


class SearchStrategyAgent:
    """Phase 1.5: reads the CandidateProfile from the blackboard and generates search queries."""

    def __init__(self, llm):
        self.llm = llm

    def generate_queries(self, profile: CandidateProfile) -> list:
        log.info("Generating job search queries based on candidate profile")
        user_content = STRATEGY_PROMPT.format(
            summary=profile.summary,
            skills=", ".join(profile.skills),
            seniority=profile.seniority,
            job_titles=", ".join(profile.job_titles),
        )
        response = self.llm.chat(
            system=STRATEGY_SYSTEM,
            user=user_content,
        )
        data = extract_json(response)
        queries = [str(q).strip() for q in data.get("search_queries", [])][:3]
        log.info("Generated queries: %s", queries)
        return queries
