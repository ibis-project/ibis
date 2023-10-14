from __future__ import annotations

import subprocess
import sys
from importlib.metadata import EntryPoint
from typing import NamedTuple

import pytest

import ibis


def test_backends_are_cached():
    assert ibis.sqlite is ibis.sqlite
    del ibis.sqlite  # delete to force recreation
    assert ibis.sqlite is ibis.sqlite


def test_backends_tab_completion():
    assert hasattr(ibis, "sqlite")
    del ibis.sqlite  # delete to ensure not real attr
    assert "sqlite" in dir(ibis)
    assert ibis.sqlite is ibis.sqlite
    assert "sqlite" in dir(ibis)  # in dir even if already created


def test_public_backend_methods():
    public = {m for m in dir(ibis.sqlite) if not m.startswith("_")}
    assert public == {"connect", "compile", "has_operation", "add_operation", "name"}


def test_missing_backend():
    msg = "module 'ibis' has no attribute 'foo'."
    with pytest.raises(AttributeError, match=msg):
        ibis.foo  # noqa: B018


def test_multiple_backends(mocker):
    class Distribution(NamedTuple):
        entry_points: list[EntryPoint]

    entrypoints = [
        EntryPoint(
            name="foo",
            value="ibis.backends.backend1",
            group="ibis.backends",
        ),
        EntryPoint(
            name="foo",
            value="ibis.backends.backend2",
            group="ibis.backends",
        ),
    ]
    if sys.version_info < (3, 10):
        return_value = {"ibis.backends": entrypoints}
    else:
        return_value = entrypoints

    mocker.patch("importlib.metadata.entry_points", return_value=return_value)

    msg = r"\d+ packages found for backend 'foo'"
    with pytest.raises(RuntimeError, match=msg):
        ibis.foo  # noqa: B018


@pytest.mark.parametrize("module", ["pandas", "pyarrow"])
def test_no_import(module):
    script = f"""
import ibis
import sys

assert "{module}" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
