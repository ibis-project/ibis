import builtins
import os

import pytest

import ibis


@pytest.fixture(autouse=True)
def add_ibis(monkeypatch, doctest_namespace):
    # disable color for doctests so we don't have to include
    # escape codes in docstrings
    monkeypatch.setitem(os.environ, "NO_COLOR", "1")
    # Explicitly set the column width
    monkeypatch.setitem(os.environ, "COLUMNS", "80")
    # reset interactive mode to False for doctests that don't execute
    # expressions
    ibis.options.interactive = False
    # workaround the fact that doctests include everything in the tested module
    # for selectors we have an `all` function
    #
    # the clash doesn't really pop up in practice, but we can rename it to
    # `all_` in 6.0 if desired
    doctest_namespace["all"] = builtins.all
