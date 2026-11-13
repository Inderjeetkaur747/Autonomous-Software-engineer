from types import SimpleNamespace

from swe_agent.config import Settings
from swe_agent.llm import GroqLLM


def make_llm(message):
    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            )

    completions = Completions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = Settings(_env_file=None, groq_api_key="gsk_test")
    return GroqLLM(settings, client=fake, sleep=lambda _s: None), completions


def test_tool_request_is_parsed_and_sent_with_tools():
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="read_file", arguments='{"path": "a.py"}')
    )
    llm, completions = make_llm(SimpleNamespace(content=None, tool_calls=[call]))
    turn = llm.chat_tools([{"role": "user", "content": "hi"}], [{"type": "function"}])

    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
        ("c1", "read_file", '{"path": "a.py"}')
    ]
    sent = completions.calls[0]
    assert sent["tool_choice"] == "auto"
    assert sent["tools"] == [{"type": "function"}]
    assert "response_format" not in sent
    assert llm.total_tokens == 10


def test_assistant_message_keeps_tool_calls_for_the_next_request():
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="grep", arguments="{}"))
    llm, _ = make_llm(SimpleNamespace(content=None, tool_calls=[call]))
    message = llm.chat_tools([], []).to_message()
    assert message["role"] == "assistant"
    assert message["content"] is None
    assert message["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "grep", "arguments": "{}"},
    }


def test_plain_answer_has_no_tool_calls():
    llm, _ = make_llm(SimpleNamespace(content="DONE"))  # no tool_calls attribute at all
    turn = llm.chat_tools([], [])
    assert turn.content == "DONE"
    assert turn.tool_calls == []
    assert "tool_calls" not in turn.to_message()