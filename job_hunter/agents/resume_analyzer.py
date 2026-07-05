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

PROFILE_PROMPT = """Analyze the resume below and extract high-quality recruiting insights. Your goal is to build a rich candidate profile that enables downstream search and vetting agents to find and grade highly relevant jobs.

Return ONLY a JSON object with these keys:
- "summary": A highly descriptive, dense 3-4 sentence professional summary. Highlight:
  1. Core engineering specializations (e.g., Generative AI/LLMs, RAG systems, Computer Vision, Python Backend).
  2. Prominent project accomplishments, model architectures built, or research experience.
  3. Key strengths and academic/professional background (e.g. IIT, research-driven).
- "skills": A comprehensive list of technical skills, libraries, frameworks, programming languages, and tools found on the resume (e.g., Python, PyTorch, TensorFlow, OpenCV, YOLO, FastAPI, LangChain, LangGraph, Git, Docker, Hugging Face). Extract every significant technical term to aid keyword-matching.
- "seniority": One of "Junior" (for freshers, interns, <2 yrs exp), "Mid-level", "Senior", "Staff/Principal".
- "job_titles": 4-5 specific target job titles, ordered by match strength, representing roles the candidate is highly qualified to apply for (e.g., "AI Engineer", "LLM Developer", "Computer Vision Engineer", "ML Engineer", "FastAPI Developer").
- "search_queries": 3 optimized job-board search query strings using Boolean logic.
  * Query 1 (GenAI/LLM): Target Generative AI, LLM, RAG, and AI Agent roles (e.g. `(AI OR LLM OR GenAI) AND (LangChain OR RAG OR Agent) AND Python`).
  * Query 2 (Vision/Deep Learning): Target Computer Vision and Deep Learning roles (e.g. `("Computer Vision" OR OpenCV OR YOLO) AND (ML OR AI OR "Deep Learning")`).
  * Query 3 (Backend/ML Engineering): Target Backend AI development and ML Engineering roles (e.g. `(Python OR FastAPI OR Backend) AND (ML OR AI OR "Machine Learning")`).
  Ensure the queries are tailored for job portal filters. If the candidate is a fresher/junior, include junior or intern query variations.

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
        log.info("Extracted %d characters from resume. Content:\n%s", len(raw), raw)
        response = self.llm.chat(
            system=PROFILE_SYSTEM,
            user=PROFILE_PROMPT.format(resume=raw[:50000]),
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
