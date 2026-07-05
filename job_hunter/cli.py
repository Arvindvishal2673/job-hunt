"""CLI entrypoint for the Resume-Driven AI Job Hunter Agent."""

import argparse
import logging
import sys

from . import config
from .models import JobSearchCriteria
from .orchestrator import ResumeJobOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-hunter",
        description="Resume-driven AI Job Hunter Agent",
    )
    parser.add_argument("--resume", required=True, help="Path to resume (.pdf, .txt, .md)")
    parser.add_argument(
        "--max-evals",
        type=int,
        default=config.MAX_EVALS_DEFAULT,
        help="Maximum number of LLM job evaluations per run (cost control)",
    )
    parser.add_argument("--remote-only", action="store_true", help="Only keep remote jobs")
    parser.add_argument(
        "--location", action="append", default=[], help="Preferred location (repeatable)"
    )
    parser.add_argument(
        "--keyword", action="append", default=[], help="Extra search keyword (repeatable)"
    )
    parser.add_argument("--min-salary", type=int, help="Minimum salary threshold")
    parser.add_argument(
        "--target-india-only", action="store_true", help="Search only Indian portals and domains"
    )
    parser.add_argument(
        "--output", default="outputs/job_matches.xlsx", help="Output .xlsx path"
    )
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    criteria = JobSearchCriteria(
        keywords=args.keyword,
        locations=args.location,
        remote_only=args.remote_only,
        min_salary=args.min_salary,
        target_india_only=args.target_india_only,
    )
    try:
        result = ResumeJobOrchestrator().run(
            args.resume, criteria, max_evals=args.max_evals, output_path=args.output
        )
    except Exception as exc:
        print(f"Job hunt failed: {exc}", file=sys.stderr)
        return 1

    metrics = result["metrics"]
    print(
        f"\nFound {metrics['total_found']} unique jobs, evaluated {metrics['evaluated']} "
        f"({metrics['strong_fits']} strong fits) in {metrics['elapsed_seconds']}s"
    )
    print(f"Results saved to {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
