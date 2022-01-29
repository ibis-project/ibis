from __future__ import annotations

import sys
from typing import NamedTuple

import pytest

import ibis
from ibis.backends.base import BaseBackend

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

EntryPoint = importlib_metadata.EntryPoint


def test_backends_are_cached():
    # can't use `hasattr` since it calls `__getattr__`
    if 'sqlite' in dir(ibis):
        del ibis.sqlite
    assert isinstance(ibis.sqlite, BaseBackend)
    assert 'sqlite' in dir(ibis)


def test_missing_backend():
    msg = "If you are trying to access the 'foo' backend"
    with pytest.raises(AttributeError, match=msg):
        ibis.foo


def test_multiple_backends(mocker):
    if sys.version_info[:2] < (3, 8):
        module = 'importlib_metadata'
    else:
        module = 'importlib.metadata'

    api = f"{module}.entry_points"

    class Distribution(NamedTuple):
        entry_points: list[EntryPoint]

    return_value = {
        "ibis.backends": [
            EntryPoint(
                name="foo",
                value='ibis.backends.backend1',
                group="ibis.backends",
            ),
            EntryPoint(
                name="foo",
                value='ibis.backends.backend2',
                group="ibis.backends",
            ),
        ],
    }

    mocker.patch(api, return_value=return_value)

    msg = r"\d+ packages found for backend 'foo'"
    with pytest.raises(RuntimeError, match=msg):
        ibis.foo
