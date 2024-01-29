from __future__ import annotations

import builtins
import os
import platform

import pytest

import ibis

SANDBOXED = (
    any(key.startswith("NIX_") for key in os.environ)
    and os.environ.get("IN_NIX_SHELL") != "impure"
)
LINUX = platform.system() == "Linux"
MACOS = platform.system() == "Darwin"
WINDOWS = platform.system() == "Windows"
CI = os.environ.get("CI") is not None


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
    # for selectors we have an `all` function as well as `ibis.range`
    #
    # the clashes don't really pop up in practice because it's unlikely for
    # people to write `from $MODULE_BEING_TESTED import *`
    doctest_namespace["all"] = builtins.all
    doctest_namespace["range"] = builtins.range


not_windows = pytest.mark.skipif(
    condition=WINDOWS,
    reason="windows prevents two connections to the same file even in the same process",
)
