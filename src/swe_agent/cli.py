"""Command line tools. Run: uv run swe-agent doctor"""

import logging
import platform
import shutil
import sys

from rich.console import Console
from rich.table import Table

from swe_agent import __version__
from swe_agent.config import get_settings
from swe_agent.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def doctor() -> int:
    """Check that the machine is ready. Returns 0 if all required checks pass."""
    console = Console()
    settings = get_settings()
    setup_logging(settings.log_level)

    checks: list[tuple[str, bool, str, bool]] = []  # (name, ok, detail, required)

    py_ok = sys.version_info >= (3, 11)
    checks.append(("Python >= 3.11", py_ok, platform.python_version(), True))

    has_groq = bool(settings.groq_api_key.get_secret_value())
    checks.append(("GROQ_API_KEY set", has_groq, "found" if has_groq else "missing", True))

    has_gh = bool(settings.github_token.get_secret_value())
    checks.append(("GITHUB_TOKEN set", has_gh, "found" if has_gh else "optional", False))

    git_path = shutil.which("git")
    checks.append(("Git installed", git_path is not None, git_path or "not found", True))

    docker_path = shutil.which("docker")
    checks.append((
        "Docker installed", 
        docker_path is not None, 
        docker_path or "needed in Phase 12", 
        False
    ))

    table = Table(title=f"swe-agent doctor (v{__version__})")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    all_required_ok = True
    for name, ok, detail, required in checks:
        if ok:
            status = "[green]OK[/green]"
        elif required:
            status = "[red]FAIL[/red]"
            all_required_ok = False
        else:
            status = "[yellow]SKIP[/yellow]"
        table.add_row(name, status, detail)

    console.print(table)
    logger.info("doctor finished, required checks ok = %s", all_required_ok)
    return 0 if all_required_ok else 1


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "doctor":
        raise SystemExit(doctor())
    print("Usage: swe-agent doctor")
    raise SystemExit(2)