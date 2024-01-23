from __future__ import annotations

import pytest

from ibis.backends.base import _get_backend_names

all_backends = list(_get_backend_names())


@pytest.mark.notimpl(all_backends)
def test_notimpl(con):
    raise Exception


@pytest.mark.notimpl(all_backends, raises=None)
def test_notimpl_raises_none(con):
    raise Exception


@pytest.mark.notimpl(all_backends, raises=(None, None))
def test_notimpl_raises_none_tuple(con):
    raise Exception


@pytest.mark.notimpl(all_backends, raises=(Exception, None))
def test_notimpl_raises_tuple_exception_none(con):
    raise Exception


@pytest.mark.notyet(all_backends)
def test_notyet(con):
    raise Exception


@pytest.mark.notyet(all_backends, raises=None)
def test_notyet_raises_none(con):
    raise Exception


@pytest.mark.notyet(all_backends, raises=(None, None))
def test_notyet_raises_none_tuple(con):
    raise Exception


@pytest.mark.notyet(all_backends, raises=(Exception, None))
def test_notyet_raises_tuple_exception_none(con):
    raise Exception


@pytest.mark.never(all_backends, reason="because I said so")
def test_never(con):
    raise Exception


@pytest.mark.never(all_backends, raises=None, reason="because I said so")
def test_never_raises_none(con):
    raise Exception


@pytest.mark.never(all_backends, raises=(None, None), reason="because I said so")
def test_never_raises_none_tuple(con):
    raise Exception


@pytest.mark.never(all_backends, raises=(Exception, None), reason="because I said so")
def test_never_raises_tuple_exception_none(con):
    raise Exception
