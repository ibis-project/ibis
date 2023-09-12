from __future__ import annotations

from collections import OrderedDict

import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.pandas import Backend
from ibis.backends.pandas.execution import execute
from ibis.backends.pandas.tests.conftest import TestConf as tm


@pytest.fixture(scope="module")
def value():
    return OrderedDict([("fruit", "pear"), ("weight", 0)])


@pytest.fixture(scope="module")
def struct_client(value):
    df = pd.DataFrame(
        {
            "s": [
                OrderedDict([("fruit", "apple"), ("weight", None)]),
                value,
                OrderedDict([("fruit", "pear"), ("weight", 1)]),
            ],
            "key": list("aab"),
            "value": [1, 2, 3],
        }
    )
    return Backend().connect({"t": df})


@pytest.fixture
def struct_table(struct_client):
    return struct_client.table(
        "t",
        schema={
            "s": dt.Struct.from_tuples([("fruit", dt.string), ("weight", dt.int8)])
        },
    )


def test_struct_field_literal(value):
    struct = ibis.literal(value)
    assert struct.type() == dt.Struct.from_tuples(
        [("fruit", dt.string), ("weight", dt.int8)]
    )

    expr = struct["fruit"]
    result = execute(expr.op())
    assert result == "pear"

    expr = struct["weight"]
    result = execute(expr.op())
    assert result == 0


def test_struct_field_series(struct_table):
    t = struct_table
    expr = t.s["fruit"]
    result = expr.execute()
    expected = pd.Series(["apple", "pear", "pear"], name="fruit")
    tm.assert_series_equal(result, expected)


def test_struct_field_series_group_by_key(struct_table):
    t = struct_table
    expr = t.group_by(t.s["fruit"]).aggregate(total=t.value.sum())
    result = expr.execute()
    expected = pd.DataFrame([("apple", 1), ("pear", 5)], columns=["fruit", "total"])
    tm.assert_frame_equal(result, expected)


def test_struct_field_series_group_by_value(struct_table):
    t = struct_table
    expr = t.group_by(t.key).aggregate(total=t.s["weight"].sum())
    result = expr.execute()
    # these are floats because we have a NULL value in the input data
    expected = pd.DataFrame([("a", 0.0), ("b", 1.0)], columns=["key", "total"])
    tm.assert_frame_equal(
        result,
        expected.assign(
            total=lambda df: df.total.astype(expr.total.type().to_pandas())
        ),
    )
