"""Resume Analyzer agent: extracts resume text and builds a CandidateProfile."""

import logging
from pathlib import Path

from ..llm import extract_json
from ..models import CandidateProfile

log = logging.getLogger(__name__)

PROFILE_SYSTEM = (
    "You are an expert technical recruiter. You read resumes and respond with "
    "only a valid JSON object, no prose."
)

PROFILE_PROMPT = """Analyze the resume below and return ONLY a JSON object with these keys:
- "summary": a 2-3 sentence professional summary of the candidate
- "skills": list of technical skills found in the resume
- "seniority": one of "Junior", "Mid-level", "Senior", "Staff/Principal"
- "job_titles": 3 job titles the candidate should target
- "search_queries": 3 optimized job-board search query strings

Resume:
\"\"\"
{resume}
\"\"\"
"""


class ResumeAnalyzer:
    """Phase 1 (Ingest): reads the resume file and writes the profile to the blackboard."""

    def __init__(self, llm):
        self.llm = llm

    def extract_text(self, path) -> str:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                raise RuntimeError(
                    "pypdf is required to read PDF resumes. Run `pip install pypdf` "
                    "or provide a .txt/.md resume instead."
                )
            reader = PdfReader(str(p))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix in (".txt", ".md"):
            return p.read_text(encoding="utf-8", errors="ignore")
        raise ValueError(f"Unsupported resume format: {suffix or p.name}")

    def analyze(self, path) -> CandidateProfile:
        raw = self.extract_text(path)
        log.info("Extracted %d characters from resume", len(raw))
        response = self.llm.chat(
            system=PROFILE_SYSTEM,
            user=PROFILE_PROMPT.format(resume=raw[:12000]),
        )
        data = extract_json(response)
        return CandidateProfile(
            summary=data.get("summary", ""),
            skills=[str(s).strip() for s in data.get("skills", [])],
            seniority=data.get("seniority", "Unknown"),
            job_titles=[str(t) for t in data.get("job_titles", [])],
            search_queries=[str(q) for q in data.get("search_queries", [])][:3],
            raw_text=raw,
        )
