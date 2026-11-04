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
    if command == "prepare":
        raise SystemExit(prepare(args[1:]))
    if command == "scan":
        raise SystemExit(scan(args[1:]))
    if command == "clean":
        raise SystemExit(clean(args[1:]))
    if command == "index":
        raise SystemExit(index(args[1:]))
    if command == "search":
        raise SystemExit(search(args[1:]))
    
    print("Usage: swe-agent [doctor | llm-test | prepare ... | clean <run_id>]")
    raise SystemExit(2)

def index(args: list[str]) -> int:
    """Embed a workspace's chunks and store them in Qdrant: swe-agent index <run_id>"""
    from swe_agent.github import WorkspaceError, WorkspaceManager
    from swe_agent.indexing import (
        IndexingError,
        QdrantStore,
        build_chunks,
        collection_for,
        get_embedder,
        index_chunks,
        load_chunks,
        save_chunks,
    )

    console = Console()
    settings = get_settings()
    setup_logging(settings.log_level)
    if len(args) != 1:
        console.print("Usage: swe-agent index <run_id>")
        return 2
    try:
        workspace = WorkspaceManager(settings).get(args[0])
    except WorkspaceError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        return 1

    chunks_file = workspace.root / "chunks.jsonl"
    if chunks_file.exists():
        chunks = load_chunks(chunks_file)
    else:
        chunks = build_chunks(workspace.repo_dir, settings).chunks
        save_chunks(chunks, chunks_file)

    console.print("Embedding chunks. The first run downloads the model (about 130 MB)...")
    embedder = get_embedder(settings)
    store = QdrantStore(settings)
    try:
        stats = index_chunks(
            chunks,
            collection=collection_for(workspace.run_id),
            embedder=embedder,
            store=store,
            batch_size=settings.embedding_batch_size,
        )
    except IndexingError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        return 1
    finally:
        store.close()

    table = Table(title="Index built")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Collection", stats.collection)
    table.add_row("Chunks stored", str(stats.chunk_count))
    table.add_row("Vector size", str(stats.vector_dim))
    table.add_row("Embedding model", stats.embedding_model)
    table.add_row("Seconds", str(stats.seconds))
    console.print(table)
    return 0


def search(args: list[str]) -> int:
    """Semantic search: swe-agent search <run_id> <question words...>"""
    from rich.markup import escape
    from rich.panel import Panel
    from rich.syntax import Syntax

    from swe_agent.github import WorkspaceError, WorkspaceManager
    from swe_agent.indexing import (
        IndexingError,
        QdrantStore,
        collection_for,
        get_embedder,
        vector_search,
    )

    console = Console()
    settings = get_settings()
    setup_logging(settings.log_level)
    if len(args) < 2:
        console.print('Usage: swe-agent search <run_id> "your question"')
        return 2
    run_id, query = args[0], " ".join(args[1:])
    try:
        workspace = WorkspaceManager(settings).get(run_id)
    except WorkspaceError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        return 1

    store = QdrantStore(settings)
    try:
        hits = vector_search(
            query,
            collection=collection_for(workspace.run_id),
            embedder=get_embedder(settings),
            store=store,
            k=settings.default_top_k,
        )
    except (IndexingError, ValueError) as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        return 1
    finally:
        store.close()

    table = Table(title=f"Top {len(hits)} results for: {escape(query)}")
    for column in ("#", "citation", "kind", "name", "score", "first line"):
        table.add_column(column)
    for rank, hit in enumerate(hits, start=1):
        first_line = next((ln.strip() for ln in hit.chunk.text.splitlines() if ln.strip()), "")
        table.add_row(
            str(rank),
            hit.chunk.citation.label,
            hit.chunk.kind.value,
            hit.chunk.name or "-",
            f"{hit.score:.3f}",
            escape(first_line[:60]),
        )
    console.print(table)

    if hits:
        best = hits[0].chunk
        syntax = Syntax(best.text, best.language, line_numbers=True, start_line=best.start_line)
        console.print(Panel(syntax, title=f"Best match: {best.citation.label}"))
    return 0

def scan(args: list[str]) -> int:
    """Scan and chunk a prepared workspace: swe-agent scan <run_id>"""
    from rich.panel import Panel
    from rich.syntax import Syntax

    from swe_agent.github import WorkspaceError, WorkspaceManager
    from swe_agent.indexing import build_chunks, save_chunks

    console = Console()
    settings = get_settings()
    setup_logging(settings.log_level)
    if len(args) != 1:
        console.print("Usage: swe-agent scan <run_id>")
        return 2
    try:
        workspace = WorkspaceManager(settings).get(args[0])
    except WorkspaceError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        return 1

    result = build_chunks(workspace.repo_dir, settings)
    out_file = workspace.root / "chunks.jsonl"
    save_chunks(result.chunks, out_file)

    table = Table(title="Scan summary")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Files kept", str(len(result.scan.files)))
    table.add_row("Chunks", str(len(result.chunks)))
    langs = ", ".join(f"{k}:{v}" for k, v in result.scan.language_counts().items())
    table.add_row("Languages", langs or "-")
    skipped = ", ".join(f"{k}:{v}" for k, v in sorted(result.skipped.items()))
    table.add_row("Skipped", skipped or "-")
    table.add_row("Hit file limit", str(result.scan.truncated))
    table.add_row("Saved to", str(out_file))
    console.print(table)

    chunk_table = Table(title="Chunks")
    for column in ("citation", "kind", "name", "lines"):
        chunk_table.add_column(column)
    for chunk in result.chunks[:15]:
        chunk_table.add_row(
            chunk.citation.label,
            chunk.kind.value,
            chunk.name or "-",
            str(chunk.end_line - chunk.start_line + 1),
        )
    console.print(chunk_table)

    for chunk in result.chunks:
        if chunk.kind.value in ("function", "method"):
            syntax = Syntax(
                chunk.text, "python", line_numbers=True, start_line=chunk.start_line
            )
            console.print(Panel(syntax, title=chunk.citation.label))
            break
    return 0

def prepare(args: list[str]) -> int:
    """Fetch an issue, clone its repo into a fresh workspace, and print a summary."""
    from rich.panel import Panel
    from rich.text import Text

    from swe_agent.github import (
        GitError,
        GitHubError,
        InvalidGitHubUrl,
        RepoTooLargeError,
        WorkspaceError,
        prepare_run,
        resolve_issue_ref,
    )

    console = Console()
    settings = get_settings()
    setup_logging(settings.log_level)

    if len(args) not in (1, 2):
        console.print("Usage: swe-agent prepare <issue-url>")
        console.print("   or: swe-agent prepare <owner/repo> <issue-number>")
        return 2

    try:
        ref = resolve_issue_ref(args[0]) if len(args) == 1 else resolve_issue_ref(args[1], args[0])
        run = prepare_run(ref, settings=settings)
    except (InvalidGitHubUrl, GitHubError, GitError, RepoTooLargeError, WorkspaceError) as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        return 1

    table = Table(title="Run prepared")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Run id", run.workspace.run_id)
    table.add_row("Repository", run.repo_info.full_name)
    table.add_row("Language", run.repo_info.language or "unknown")
    table.add_row("GitHub size (KB)", str(run.repo_info.size_kb))
    table.add_row("Issue", f"#{run.issue.number} {run.issue.title}")
    table.add_row("Labels", ", ".join(run.issue.labels) or "-")
    table.add_row("Base commit", run.head_sha[:12])
    table.add_row("Work branch", run.branch)
    table.add_row("Repo folder", str(run.workspace.repo_dir))
    console.print(table)
    # Text() shows the issue body as plain text. Rich would treat [brackets] as styling.
    console.print(Panel(Text(run.issue.body[:600] or "(empty issue body)"), title="Issue body"))
    return 0


def clean(args: list[str]) -> int:
    """Delete one workspace: swe-agent clean <run_id>"""
    from swe_agent.github import WorkspaceError, WorkspaceManager

    console = Console()
    settings = get_settings()
    setup_logging(settings.log_level)
    if len(args) != 1:
        console.print("Usage: swe-agent clean <run_id>")
        return 2
    try:
        WorkspaceManager(settings).delete(args[0])
    except WorkspaceError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        return 1

    # Also remove this run's vectors (best effort).
    try:
        from swe_agent.indexing import QdrantStore, collection_for

        store = QdrantStore(settings)
        try:
            store.delete_collection(collection_for(args[0]))
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not delete vectors: %s", exc)

    console.print(f"[green]Deleted workspace {args[0]}[/green]")
    return 0

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