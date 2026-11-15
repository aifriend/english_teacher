"""Routing tests: each public method must reach the intended service method.

The three service classes are replaced with recording fakes, so nothing
downloads a model or calls an LLM.
"""
import pytest

from english_improvement_agent import EnglishImprovementAgent as agent_module
from english_improvement_agent import EnglishImprovementAgentStyle as style_module


class Recorder:
    """Fake service: any method call returns 'name:text' and is recorded."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(text):
            self.calls.append(name)
            return f"{name}:{text}"
        return method


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(agent_module, "EnglishImprovementAgentWrite", Recorder)
    monkeypatch.setattr(agent_module, "EnglishImprovementAgentStyle", Recorder)
    monkeypatch.setattr(agent_module, "EnglishImprovementAgentSummary", Recorder)
    return agent_module.EnglishImprovementAgent()


@pytest.mark.parametrize("method, expected", [
    ("write_properly", "write_properly_vannity"),
    ("write_properly_0", "write_properly_vannity"),
    ("write_properly_1", "write_properly_coedit"),
    ("write_properly_2", "write_properly_instruct"),
    ("write_the_same_grammar_fixed", "write_the_same_grammar_fixed_with_llama_index_rag"),
    ("write_the_same_grammar_fixed_0", "write_the_same_grammar_fixed_with_llama_index_rag"),
    ("write_the_same_grammar_fixed_1", "write_the_same_grammar_fixed_with_llm_instruction"),
    ("summarize", "summarization_with_t5"),
])
def test_agent_routes_to_expected_service_method(agent, method, expected):
    assert getattr(agent, method)("hello") == f"{expected}:hello"


def test_style_service_builds_backends_lazily(monkeypatch):
    built = []

    class FakeLlama(Recorder):
        def __init__(self):
            super().__init__()
            built.append("llama")

        def write_the_same_grammar_fixed(self, text):
            return f"llama:{text}"

    class FakeInstruct(Recorder):
        def __init__(self):
            super().__init__()
            built.append("instruct")

        def write_the_same_grammar_fixed(self, text):
            return f"instruct:{text}"

    monkeypatch.setattr(style_module, "WriteStyleLlamaIndex", FakeLlama)
    monkeypatch.setattr(style_module, "WriteProperlyInstruct", FakeInstruct)
    style = style_module.EnglishImprovementAgentStyle()
    assert built == []

    assert style.write_the_same_grammar_fixed_with_llama_index_rag("a") == "llama:a"
    assert style.write_the_same_grammar_fixed_with_llama_index_rag("b") == "llama:b"
    assert built == ["llama"]  # built once, then reused

    assert style.write_the_same_grammar_fixed_with_llm_instruction("c") == "instruct:c"
    assert built == ["llama", "instruct"]
