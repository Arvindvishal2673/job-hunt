# Resume-Driven AI Job Hunter Agent

A production-grade, local AI agent that automates job hunting:

1. Parses a candidate's resume (PDF, TXT, Markdown).
2. Uses a Groq-hosted LLM to generate an optimized search strategy.
3. Queries **15 job boards** (via DuckDuckGo `site:` search) plus 3 public job APIs (Remotive, RemoteOK, Arbeitnow) in parallel.
4. Uses an LLM "brain" to vet every listing: fit score, fit decision, match reasons, and gap analysis.
5. Exports results to a styled Excel spreadsheet with conditional color highlights.

## Architecture

A lightweight, custom agentic framework (no LangChain/CrewAI) built on native Python:

- **Blackboard communication pattern**: agents share state through the central `ResumeJobOrchestrator`.
- **Structured interface invariance**: all source agents inherit from the abstract `JobSourceAgent`.
- **State-sharing contracts**: immutable dataclasses (`JobSearchCriteria`, `CandidateProfile`, `JobListing`).
- **I/O concurrency**: `ThreadPoolExecutor` for parallel network requests and LLM evaluations.

### Agents

| Agent | Module | Role |
| :--- | :--- | :--- |
| Resume Analyzer | `job_hunter/agents/resume_analyzer.py` | Extracts resume text, builds `CandidateProfile` + search queries via LLM |
| Platform Searcher | `job_hunter/agents/platform_searcher.py` | DuckDuckGo `site:` search across 15 platforms in 3 parallel groups |
| API Agents | `job_hunter/agents/api_agents.py` | Remotive, RemoteOK, Arbeitnow JSON feeds, normalized to `JobListing` |
| Vetting Brain | `job_hunter/agents/vetting.py` | LLM hiring-manager evaluation: fit score, decision, reasons, gaps |
| Orchestrator | `job_hunter/orchestrator.py` | Blackboard state, dedupe, pre-filter, parallel vetting, reporting |

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                              # then add your GROQ_API_KEY
```

Get a free API key at https://console.groq.com.

## Usage

```bash
python -m job_hunter.cli --resume path/to/resume.pdf

# With options
python -m job_hunter.cli --resume resume.pdf --remote-only --max-evals 30 \
    --location "Berlin" --keyword "python" --output outputs/job_matches.xlsx
```

Output: `outputs/job_matches.xlsx` with a color-coded **Job Matches** sheet (green = Strong Fit, yellow = Decent Fit, red = Weak Fit) and a **Candidate Profile** summary sheet.

## Cost & performance

- 1 resume-analysis LLM call + at most `--max-evals` (default 40) evaluation calls per run.
- All source queries and evaluations run in parallel thread pools; a typical run finishes in under 60 seconds.

## Tests

Fully offline, with mocked LLM and network calls:

```bash
pytest tests/ -v
```
