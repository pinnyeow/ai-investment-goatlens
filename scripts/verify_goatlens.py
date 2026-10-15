#!/usr/bin/env python3
"""Run GOATlens deterministic pre-PR verification."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
PYTHON = BACKEND / "venv" / "bin" / "python"
RUFF = BACKEND / "venv" / "bin" / "ruff"


def run(label: str, command: list[str], cwd: Path) -> None:
    print(f"\n== {label} ==")
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    if not PYTHON.exists() or not RUFF.exists():
        print("backend/venv with pytest and ruff is required", file=sys.stderr)
        return 2

    checks = [
        ("Backend tests", [str(PYTHON), "-m", "pytest", "-q"], BACKEND),
        ("Python lint", [str(RUFF), "check", "."], BACKEND),
        ("Whitespace and conflict markers", ["git", "diff", "--check"], ROOT),
        (
            "Workflow syntax",
            [
                str(PYTHON),
                "-c",
                (
                    "import pathlib,yaml; "
                    "yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); "
                    "yaml.safe_load(pathlib.Path('.github/workflows/smoke-tests.yml').read_text())"
                ),
            ],
            ROOT,
        ),
    ]

    try:
        for label, command, cwd in checks:
            run(label, command, cwd)
    except subprocess.CalledProcessError as error:
        print(f"\nFAILED: {error.cmd} exited {error.returncode}", file=sys.stderr)
        return error.returncode

    print("\nGOATlens deterministic verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
