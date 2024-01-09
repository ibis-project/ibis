from __future__ import annotations

import re

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis import udf


def test_multiple_calls_redefinition():
    @udf.scalar.python
    def my_len(s: str) -> float:
        return s.length

    s = ibis.literal("abcd")
    expr = my_len(s) + my_len(s)

    @udf.scalar.python
    def my_len(s: str) -> float:
        return s.length + 1

    expr = expr + my_len(s)

    sql = ibis.bigquery.compile(expr)
    assert len(set(re.findall(r"my_len_(\d+)", sql))) == 2


@pytest.mark.parametrize("determinism", [True, False, None])
def test_udf_determinism(determinism):
    @udf.scalar.python(determinism=determinism)
    def my_len(s: str) -> float:
        return s.length

    s = ibis.literal("abcd")
    expr = my_len(s)

    sql = ibis.bigquery.compile(expr)

    if not determinism:
        assert "NOT DETERMINISTIC" in sql
    else:
        assert "DETERMINISTIC" in sql and "NOT DETERMINISTIC" not in sql


@pytest.mark.parametrize(
    ("argument_type", "return_type"),
    [
        # invalid input type
        (dt.int64, dt.float64),
        # invalid return type
        (dt.float64, dt.int64),
        # complex argument type, valid return type
        (dt.Array(dt.int64), dt.float64),
        # valid argument type, complex invalid return type
        (dt.float64, dt.Array(dt.int64)),
        # both invalid
        (dt.Array(dt.Array(dt.int64)), dt.int64),
        # struct type with nested integer, valid return type
        (dt.Struct({"x": dt.Array(dt.int64)}), dt.float64),
    ],
    ids=str,
)
def test_udf_int64(argument_type, return_type):
    # invalid argument type, valid return type
    @udf.scalar.python(signature=((argument_type,), return_type))
    def my_func(x):
        return 1

    expr = my_func(None)
    with pytest.raises(com.UnsupportedBackendType):
        ibis.bigquery.compile(expr)
