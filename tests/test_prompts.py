"""Prompt tests: grammar-only vs grammar-and-style prompts must not be swapped."""
from types import SimpleNamespace

from english_improvement_agent.WriteProperlyInstruct import (
    GRAMMAR_ONLY_SYSTEM_PROMPT,
    STYLE_SYSTEM_PROMPT,
    WriteProperlyInstruct,
)
from english_improvement_agent.WriteStyleLlamaIndex import WriteStyleLlamaIndex


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="fixed text")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_instruct(completions):
    """WriteProperlyInstruct without running __init__ (which would build an OpenAI client)."""
    obj = object.__new__(WriteProperlyInstruct)
    obj.logger = SimpleNamespace(Information=lambda *_a, **_k: None)
    obj.model = SimpleNamespace(
        model_name="fake-model",
        llm=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        get_doc_from_langchain=lambda ds_path: [],
    )
    return obj


def test_system_prompts_have_distinct_roles():
    assert "focus solely on correcting grammatical errors" in GRAMMAR_ONLY_SYSTEM_PROMPT
    assert "Avoid making stylistic changes" in GRAMMAR_ONLY_SYSTEM_PROMPT
    assert "sophisticated writing style" in STYLE_SYSTEM_PROMPT
    assert "Avoid making stylistic changes" not in STYLE_SYSTEM_PROMPT


def test_write_properly_uses_grammar_and_style_prompt():
    completions = FakeCompletions()
    result = make_instruct(completions).write_properly("This sentences are bad")
    system, user = completions.kwargs["messages"]
    assert result == "fixed text"
    assert system["content"] == STYLE_SYSTEM_PROMPT
    assert "This sentences are bad" in user["content"]
    assert "do not change the style" not in user["content"]


def test_write_the_same_grammar_fixed_uses_grammar_only_prompt(monkeypatch):
    from english_improvement_agent import WriteProperlyInstruct as module

    doc = SimpleNamespace(page_content="EXAMPLE CONTEXT")
    fake_retriever = SimpleNamespace(invoke=lambda text: [doc])
    fake_db = SimpleNamespace(as_retriever=lambda: fake_retriever)
    monkeypatch.setattr(module, "Chroma", SimpleNamespace(from_documents=lambda docs, emb: fake_db))
    monkeypatch.setattr(module, "OpenAIEmbeddings", lambda: None)

    completions = FakeCompletions()
    result = make_instruct(completions).write_the_same_grammar_fixed("He go home")
    system, user = completions.kwargs["messages"]
    assert result == "fixed text"
    assert system["content"] == GRAMMAR_ONLY_SYSTEM_PROMPT
    assert "He go home" in user["content"]
    assert "EXAMPLE CONTEXT" in user["content"]
    assert "do not change the style" in user["content"]


def test_llama_index_prompt_is_strict_grammar_only():
    captured = {}
    obj = object.__new__(WriteStyleLlamaIndex)
    obj.query_engine = SimpleNamespace(query=lambda q: captured.setdefault("q", q) and "ok")

    assert obj.write_the_same_grammar_fixed("She don't know") == "ok"
    question = captured["q"]
    assert "She don't know" in question
    assert "ONLY the grammatical errors" in question
    assert "Return only the corrected sentence" in question
    for forbidden in ("sophisticated", "Feel free to rephrase", "multiples outputs"):
        assert forbidden not in question
