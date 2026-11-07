from __future__ import annotations

import builtins
import importlib.metadata
import os
import platform
from pathlib import Path

import pytest
from packaging.version import Version

import ibis

SANDBOXED = (
    any(key.startswith("NIX_") for key in os.environ)
    and os.environ.get("IN_NIX_SHELL") != "impure"
)
LINUX = platform.system() == "Linux"
MACOS = platform.system() == "Darwin"
WINDOWS = platform.system() == "Windows"
ARM64 = platform.machine() in ("arm64", "aarch64")
CI = os.environ.get("CI") is not None
SPARK_REMOTE = os.environ.get("SPARK_REMOTE")
IS_SPARK_REMOTE = bool(SPARK_REMOTE)
# the flink extra doesn't depend on apache-flink (see pyproject.toml), so look
# the installed version up; None when PyFlink isn't installed
try:
    PYFLINK_VERSION = Version(importlib.metadata.version("apache-flink"))
except importlib.metadata.PackageNotFoundError:
    PYFLINK_VERSION = None
IS_PYFLINK_1X = PYFLINK_VERSION is not None and PYFLINK_VERSION.major == 1
# python that flink's taskmanager runs UDFs with: the version shipped by the
# ubuntu release the flink image is built on (see docker/flink/Dockerfile)
FLINK_WORKER_PYTHON_VERSION = (3, 10) if IS_PYFLINK_1X else (3, 12)


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


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the test data directory.

    Returns
    -------
    Path
        Test data directory
    """
    root = Path(__file__).absolute().parents[1]

    return root / "ci" / "ibis-testing-data"
