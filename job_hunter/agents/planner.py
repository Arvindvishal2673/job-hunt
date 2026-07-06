"""PlannerAgent: uses Groq tool-calling to autonomously select job source agents.

This is the first genuinely agentic component in the pipeline. Instead of
always querying every source, the LLM is given a menu of tools (one per
source agent) and decides which ones are most relevant for this candidate.
"""

import logging
from typing import List

from ..models import CandidateProfile
from .base import JobSourceAgent

log = logging.getLogger(__name__)

# System prompt that turns the LLM into a planning agent
PLANNER_SYSTEM = """You are an intelligent job search planning agent.

Your job is to select the best combination of job source APIs to query for
a specific candidate. You must call the tools that are most likely to return
high-quality, relevant results for this candidate's profile and location.

Rules:
- Always call at least 2 tools.
- For India-based candidates, prioritize LinkedIn (Apify) and Adzuna India.
- For ML/AI candidates, always include LinkedIn as it has the best ML listings.
- For remote roles, prefer Remotive and RemoteOK.
- For recent graduates or interns, include DirectATS for company career pages.
- You may call all tools if the profile is broad or general.
"""

# Tool schema for each available source agent (OpenAI function-calling format)
SOURCE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "activate_apify_linkedin",
            "description": (
                "Scrapes LinkedIn job listings via Apify cloud actors. Best for "
                "ML, AI, Data Science, Software Engineering roles. Essential for "
                "India-based candidates and any technical roles."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_adzuna",
            "description": (
                "Queries the Adzuna job board REST API. Strong coverage of "
                "India tech jobs, mid-level engineering, and data roles. "
                "Good for both India and global positions."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_direct_ats",
            "description": (
                "Scrapes corporate ATS systems directly (Greenhouse, Lever). "
                "Best for top-tier tech companies (startups to FAANG). "
                "Ideal for recent graduates targeting product companies."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_remotive",
            "description": (
                "Queries the Remotive API for remote-first job listings. "
                "Best for candidates seeking fully remote roles globally, "
                "especially in software engineering, ML, and data science."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_remoteok",
            "description": (
                "Queries the RemoteOK API for remote tech jobs. "
                "Good supplement to Remotive for remote roles, "
                "particularly strong for developer and ML positions."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_arbeitnow",
            "description": (
                "Queries the Arbeitnow API for European tech jobs with visa "
                "sponsorship. Best for candidates open to relocating to Europe "
                "or seeking international opportunities."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Maps tool function names to their source agent class names
TOOL_NAME_TO_AGENT_CLASS = {
    "activate_apify_linkedin": "ApifyLinkedInAgent",
    "activate_adzuna": "AdzunaAgent",
    "activate_direct_ats": "DirectATSAgent",
    "activate_remotive": "RemotiveAgent",
    "activate_remoteok": "RemoteOKAgent",
    "activate_arbeitnow": "ArbeitnowAgent",
}


class PlannerAgent:
    """Phase 2a: uses LLM tool-calling to autonomously select which job sources to query.

    The LLM receives the candidate profile and a list of tool schemas
    (one per source agent). It autonomously selects which sources to activate
    based on the candidate's skills, location preference, and seniority.
    """

    def __init__(self, llm):
        self.llm = llm

    def select_sources(
        self,
        profile: CandidateProfile,
        all_sources: List[JobSourceAgent],
        target_india_only: bool = False,
    ) -> List[JobSourceAgent]:
        """Ask the LLM which sources to activate. Returns a filtered source list.

        Falls back to using all provided sources if tool-calling fails.
        """
        # Build the user message with full candidate context
        location_hint = "India only" if target_india_only else "global / remote"
        user_msg = f"""Candidate Profile:
- Summary: {profile.summary}
- Skills: {', '.join(profile.skills[:15])}
- Seniority: {profile.seniority}
- Target Job Titles: {', '.join(profile.job_titles)}
- Search Queries: {', '.join(profile.search_queries)}
- Location Preference: {location_hint}

Select the best combination of job source tools for this candidate."""

        try:
            tool_calls = self.llm.chat_with_tools(
                system=PLANNER_SYSTEM,
                user=user_msg,
                tools=SOURCE_TOOL_SCHEMAS,
                temperature=0.0,
            )
        except Exception as exc:
            log.warning("PlannerAgent tool-calling failed: %s. Using all sources.", exc)
            return all_sources

        if not tool_calls:
            log.warning("PlannerAgent: LLM returned no tool calls. Using all sources.")
            return all_sources

        # Extract which tool names the LLM selected
        selected_names = {tc["function"]["name"] for tc in tool_calls}
        selected_class_names = {
            TOOL_NAME_TO_AGENT_CLASS[name]
            for name in selected_names
            if name in TOOL_NAME_TO_AGENT_CLASS
        }

        log.info("PlannerAgent selected sources: %s", sorted(selected_class_names))

        # Filter all_sources to only those the LLM selected
        selected = [s for s in all_sources if type(s).__name__ in selected_class_names]

        # Safety: always return at least something
        return selected if selected else all_sources
