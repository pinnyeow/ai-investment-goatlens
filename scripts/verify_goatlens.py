#!/usr/bin/env python3
"""Run GOATlens deterministic pre-PR verification from the repository root."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
AFFECTED_TESTS = (
    "backend/tests/test_sec_market_split.py",
    "backend/tests/test_consensus_integrity.py",
    "backend/tests/test_provider_architecture.py",
)


def run(label: str, command: list[str]) -> None:
    print(f"\n== {label} ==")
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def candidate_diff_range() -> list[str]:
    """Return the committed candidate range, including every PR commit."""
    if os.environ.get("GITHUB_EVENT_NAME") == "push":
        return ["HEAD^", "HEAD"]
    base_branch = os.environ.get("GITHUB_BASE_REF", "main")
    base_ref = f"origin/{base_branch}"
    merge_base = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not merge_base:
        raise RuntimeError("Unable to determine candidate diff merge base")
    return [f"{merge_base}..HEAD"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--affected-only",
        action="store_true",
        help="Run the optimized validation-sensitive suite without repeating full lint/gates.",
    )
    args = parser.parse_args()

    if args.affected_only:
        checks = [
            (
                "Optimized affected validation suite",
                [PYTHON, "-m", "pytest", "-q", *AFFECTED_TESTS],
            )
        ]
    else:
        checks = [
            ("Backend tests", [PYTHON, "-m", "pytest", "-q", "backend/tests"]),
            ("Python lint", [PYTHON, "-m", "ruff", "check", "backend"]),
            (
                "Committed candidate whitespace and conflict markers",
                ["git", "diff", "--check", *candidate_diff_range()],
            ),
            ("Working tree whitespace and conflict markers", ["git", "diff", "--check"]),
            (
                "Workflow syntax",
                [
                    PYTHON,
                    "-c",
                    (
                        "import pathlib,yaml; "
                        "yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); "
                        "yaml.safe_load(pathlib.Path('.github/workflows/smoke-tests.yml').read_text())"
                    ),
                ],
            ),
        ]

    try:
        for label, command in checks:
            run(label, command)
    except subprocess.CalledProcessError as error:
        print(f"\nFAILED: {error.cmd} exited {error.returncode}", file=sys.stderr)
        return error.returncode

    print("\nGOATlens deterministic verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
