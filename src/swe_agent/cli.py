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
    command = args[0] if args else ""
    if command == "doctor":
        raise SystemExit(doctor())
    if command == "llm-test":
        raise SystemExit(llm_test())
    print("Usage: swe-agent [doctor | llm-test]")
    raise SystemExit(2)


_SAMPLE_TITLE = "Login fails when the email has capital letters"
_SAMPLE_BODY = (
    "A user registered with 'john@example.com'. When they log in with 'John@Example.com' "
    "the app says the user does not exist. Emails should be case-insensitive."
)
_SAMPLE_CODE = """\
# src/auth/login.py (lines 10-22)
def find_user(email):
    return db.query(User).filter(User.email == email).first()

def login(email, password):
    user = find_user(email)
    if user is None:
        raise AuthError("user does not exist")
    return check_password(user, password)
"""


def llm_test() -> int:
    """Send a sample issue to Groq and print the structured results."""
    console = Console()
    settings = get_settings()
    setup_logging(settings.log_level)

    if not settings.groq_api_key.get_secret_value():
        console.print("[red]GROQ_API_KEY is missing. Fix your .env first.[/red]")
        return 1

    # Imported here so `doctor` stays fast and works even if these have problems.
    from swe_agent.llm import GroqLLM, LLMError
    from swe_agent.llm.prompts import build_analysis_prompts, build_plan_prompts
    from swe_agent.schemas import ImplementationPlan, IssueAnalysis

    llm = GroqLLM(settings)
    try:
        system, user = build_analysis_prompts(_SAMPLE_TITLE, _SAMPLE_BODY, _SAMPLE_CODE)
        analysis = llm.generate_structured(
            IssueAnalysis, system=system, user=user, purpose="analysis"
        )
        console.rule("Issue analysis")
        console.print_json(analysis.model_dump_json())

        system, user = build_plan_prompts(
            _SAMPLE_TITLE, _SAMPLE_BODY, analysis.model_dump_json(), _SAMPLE_CODE
        )
        plan = llm.generate_structured(
            ImplementationPlan, system=system, user=user, purpose="plan"
        )
        console.rule("Implementation plan")
        console.print_json(plan.model_dump_json())
    except LLMError as exc:
        console.print(f"[red]LLM error:[/red] {exc}")
        return 1

    table = Table(title="LLM calls")
    for column in ("purpose", "attempt", "latency ms", "prompt tok", "completion tok"):
        table.add_column(column)
    for call in llm.calls:
        table.add_row(
            call.purpose,
            str(call.attempt),
            str(call.latency_ms),
            str(call.prompt_tokens),
            str(call.completion_tokens),
        )
    console.print(table)
    return 0