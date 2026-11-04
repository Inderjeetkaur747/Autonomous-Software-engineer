from swe_agent.indexing import chunk_file
from swe_agent.retrieval import Bm25Index

FILES = {
    "mathkit/stats.py": (
        "def average(numbers):\n    return sum(numbers) / len(numbers)\n\n\n"
        "def median(values):\n    ordered = sorted(values)\n    return ordered[len(ordered) // 2]\n"
    ),
    "app/users.py": "def get_user_by_email(email):\n    return db.find(email)\n",
    "app/net.py": "def connect_database(url):\n    return open_socket(url)\n",
}


def make_chunks(files=FILES):
    chunks = []
    for path, source in files.items():
        chunks += chunk_file(path, source, "python")
    return chunks


def test_exact_identifier_ranks_first():
    index = Bm25Index(make_chunks())
    hits = index.search("get_user_by_email")
    assert hits[0].chunk.citation.label == "app/users.py:1-2"
    assert hits[0].source == "bm25"


def test_split_words_match_snake_case_names():
    hits = Bm25Index(make_chunks()).search("user email")
    assert hits[0].chunk.name == "get_user_by_email"


def test_scores_are_positive_and_sorted():
    hits = Bm25Index(make_chunks()).search("average numbers values")
    scores = [h.score for h in hits]
    assert scores and all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_unknown_words_give_no_results():
    assert Bm25Index(make_chunks()).search("zzzxqv") == []


def test_empty_index_and_empty_query_are_safe():
    assert Bm25Index([]).search("anything") == []
    assert Bm25Index(make_chunks()).search("   ") == []


def test_k_limits_results():
    assert len(Bm25Index(make_chunks()).search("def return", k=2)) <= 2


def test_rare_word_beats_common_word():
    files = {
        "a.py": "def one():\n    common()\n    rare()\n",
        "b.py": "def two():\n    common()\n",
        "c.py": "def three():\n    common()\n",
    }
    hits = Bm25Index(make_chunks(files)).search("common rare")
    assert hits[0].chunk.file_path == "a.py"