from __future__ import annotations

from collections import OrderedDict

import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt

dd = pytest.importorskip("dask.dataframe")

from dask.dataframe.utils import tm  # noqa: E402

from ibis.backends.dask.execution import execute  # noqa: E402


@pytest.fixture(scope="module")
def value():
    return OrderedDict([("fruit", "pear"), ("weight", 0)])


@pytest.fixture(scope="module")
def struct_client(value, npartitions):
    df = dd.from_pandas(
        pd.DataFrame(
            {
                "s": [
                    OrderedDict([("fruit", "apple"), ("weight", None)]),
                    value,
                    OrderedDict([("fruit", "pear"), ("weight", 1)]),
                ],
                "key": list("aab"),
                "value": [1, 2, 3],
            }
        ),
        npartitions=npartitions,
    )
    return ibis.dask.connect({"t": df})


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
    result = expr.compile()
    expected = dd.from_pandas(
        pd.Series(["apple", "pear", "pear"], name="fruit"),
        npartitions=1,
    )
    tm.assert_series_equal(result.compute(), expected.compute(), check_index=False)


def test_struct_field_series_group_by_key(struct_table):
    t = struct_table
    expr = t.group_by(t.s["fruit"]).aggregate(total=t.value.sum())
    result = expr.compile()
    expected = dd.from_pandas(
        pd.DataFrame([("apple", 1), ("pear", 5)], columns=["fruit", "total"]),
        npartitions=1,
    )
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )


def test_struct_field_series_group_by_value(struct_table):
    t = struct_table
    expr = t.group_by(t.key).aggregate(total=t.s["weight"].sum())
    result = expr.compile()
    # these are floats because we have a NULL value in the input data
    expected = dd.from_pandas(
        pd.DataFrame([("a", 0.0), ("b", 1.0)], columns=["key", "total"]),
        npartitions=1,
    )
    tm.assert_frame_equal(
        result.compute().reset_index(drop=True),
        expected.compute().reset_index(drop=True),
    )
