import os

import pytest

import ibis


@pytest.fixture(autouse=True)
@pytest.mark.usefixtures("doctest_namespace")
def add_ibis(monkeypatch):
    # disable color for doctests so we don't have to include
    # escape codes in docstrings
    monkeypatch.setitem(os.environ, "NO_COLOR", "1")
    # reset interactive mode to False for doctests that don't execute
    # expressions
    ibis.options.interactive = False
