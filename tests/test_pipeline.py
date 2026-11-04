from swe_agent.config import Settings
from swe_agent.indexing import build_chunks, load_chunks, save_chunks


def test_pipeline_skips_binary_minified_and_secret_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "node_modules").mkdir(parents=True)
    (repo / "good.py").parent.mkdir(exist_ok=True)
    (repo / "good.py").write_text("def ok():\n    return 1\n")
    (repo / "fake.py").write_bytes(b"\x00\x01\x02 not really python")
    (repo / "bundle.js").write_text("var a=" + "1," * 1000 + "0;")
    (repo / ".env").write_text("KEY=1\n")
    (repo / "node_modules" / "x.js").write_text("x = 1\n")

    result = build_chunks(repo, Settings(_env_file=None))
    assert {c.file_path for c in result.chunks} == {"good.py"}
    assert result.skipped["binary"] == 1
    assert result.skipped["minified_or_generated"] == 1
    assert result.skipped["sensitive_name"] == 1


def test_save_and_load_roundtrip(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def a():\n    return 1\n")
    result = build_chunks(repo, Settings(_env_file=None))
    out = tmp_path / "chunks.jsonl"
    save_chunks(result.chunks, out)
    assert load_chunks(out) == result.chunks