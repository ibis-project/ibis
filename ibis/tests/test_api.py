from __future__ import annotations

import subprocess
import sys
from importlib.metadata import EntryPoint

import pytest
from pytest import param

import ibis

# FIXME(kszucs): the following backends require the sqlite backend loaded
# def test_backends_are_cached():
#     assert ibis.sqlite is ibis.sqlite
#     del ibis.sqlite  # delete to force recreation
#     assert ibis.sqlite is ibis.sqlite


# def test_backends_tab_completion():
#     assert hasattr(ibis, "sqlite")
#     del ibis.sqlite  # delete to ensure not real attr
#     assert "sqlite" in dir(ibis)
#     assert ibis.sqlite is ibis.sqlite
#     assert "sqlite" in dir(ibis)  # in dir even if already created


# def test_public_backend_methods():
#     public = {m for m in dir(ibis.sqlite) if not m.startswith("_")}
#     assert public == {"connect", "compile", "has_operation", "name"}


def test_missing_backend():
    msg = "module 'ibis' has no attribute 'foo'."
    with pytest.raises(AttributeError, match=msg):
        ibis.foo  # noqa: B018


def test_multiple_backends(mocker):
    return_value = [
        EntryPoint(name="foo", value="ibis.backends.backend1", group="ibis.backends"),
        EntryPoint(name="foo", value="ibis.backends.backend2", group="ibis.backends"),
    ]

    mocker.patch("ibis.util.backend_entry_points", return_value=return_value)

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


def test_ibis_na_deprecation_warning():
    with pytest.warns(
        DeprecationWarning, match="The 'ibis.NA' constant is deprecated as of v9.1"
    ):
        assert ibis.NA is ibis.null()


@pytest.mark.parametrize(
    "op, api_func",
    [
        param("desc", ibis.desc),
        param("asc", ibis.asc),
    ],
)
def test_ibis_desc_asc_default(op, api_func):
    t = ibis.table(schema={"a": "int", "b": "str"})

    expr = t.order_by(getattr(t["b"], op)())
    expr_api = t.order_by(api_func("b"))

    assert expr.op() == expr_api.op()


@pytest.mark.parametrize(
    "op, api_func, nulls_first",
    [
        param("desc", ibis.desc, True),
        param("asc", ibis.asc, True),
        param("desc", ibis.desc, False),
        param("asc", ibis.asc, False),
    ],
)
def test_ibis_desc_asc(op, api_func, nulls_first):
    t = ibis.table(schema={"a": "int", "b": "str"})

    expr = t.order_by(getattr(t["b"], op)(nulls_first=nulls_first))
    expr_api = t.order_by(api_func("b", nulls_first=nulls_first))

    assert expr.op() == expr_api.op()
