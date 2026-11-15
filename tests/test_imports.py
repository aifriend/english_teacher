"""Every module must import cleanly with only the declared dependencies installed."""
import importlib
import pkgutil

import pytest

import english_improvement_agent

MODULES = sorted(m.name for m in pkgutil.iter_modules(english_improvement_agent.__path__))


def test_modules_discovered():
    assert len(MODULES) >= 14


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"english_improvement_agent.{name}")
