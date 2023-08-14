from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas.testing as tm
import pyarrow as pa
import pytest

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.common.exceptions import IntegrityError
from ibis.common.grounds import Annotable
from ibis.common.patterns import CoercedTo

has_pandas = False
with contextlib.suppress(ImportError):
    import pandas as pd

    has_pandas = True

has_dask = False
with contextlib.suppress(ImportError):
    import dask.dataframe as dd  # noqa: F401

    has_dask = True


def test_whole_schema():
    schema = {
        "cid": "int64",
        "mktsegment": "string",
        "address": """struct<city: string,
                             street: string,
                             street_number: int32,
                             zip: int16>""",
        "phone_numbers": "array<string>",
        "orders": """array<struct<oid: int64,
                                  status: string,
                                  totalprice: decimal(12, 2),
                                  order_date: string,
                                  items: array<struct<iid: int64,
                                                      name: string,
                                                      price: decimal(12, 2),
                                                      discount_perc: decimal(12, 2),
                                                      shipdate: string>>>>
                                                      """,
        "web_visits": """map<string, struct<user_agent: string,
                                            client_ip: string,
                                            visit_date: string,
                                            duration_ms: int32>>""",
        "support_calls": """array<struct<agent_id: int64,
                                         call_date: string,
                                         duration_ms: int64,
                                         issue_resolved: boolean,
                                         agent_comment: string>>""",
    }
    expected = {
        "cid": dt.int64,
        "mktsegment": dt.string,
        "address": dt.Struct(
            {
                "city": dt.string,
                "street": dt.string,
                "street_number": dt.int32,
                "zip": dt.int16,
            }
        ),
        "phone_numbers": dt.Array(dt.string),
        "orders": dt.Array(
            dt.Struct(
                {
                    "oid": dt.int64,
                    "status": dt.string,
                    "totalprice": dt.Decimal(12, 2),
                    "order_date": dt.string,
                    "items": dt.Array(
                        dt.Struct(
                            {
                                "iid": dt.int64,
                                "name": dt.string,
                                "price": dt.Decimal(12, 2),
                                "discount_perc": dt.Decimal(12, 2),
                                "shipdate": dt.string,
                            }
                        )
                    ),
                }
            )
        ),
        "web_visits": dt.Map(
            dt.string,
            dt.Struct(
                {
                    "user_agent": dt.string,
                    "client_ip": dt.string,
                    "visit_date": dt.string,
                    "duration_ms": dt.int32,
                }
            ),
        ),
        "support_calls": dt.Array(
            dt.Struct(
                {
                    "agent_id": dt.int64,
                    "call_date": dt.string,
                    "duration_ms": dt.int64,
                    "issue_resolved": dt.boolean,
                    "agent_comment": dt.string,
                }
            )
        ),
    }
    assert sch.schema(schema) == sch.schema(expected)


def test_schema_from_tuples():
    schema = sch.Schema.from_tuples(
        [
            ("a", "int64"),
            ("b", "string"),
            ("c", "double"),
            ("d", "boolean"),
        ]
    )
    expected = sch.Schema(
        {"a": dt.int64, "b": dt.string, "c": dt.double, "d": dt.boolean}
    )

    assert schema == expected

    # test that duplicate field names are prohibited
    with pytest.raises(IntegrityError):
        sch.Schema.from_tuples([("a", "int64"), ("a", "string")])


def test_schema_subset():
    s1 = sch.schema([("a", dt.int64), ("b", dt.int32), ("c", dt.string)])
    s2 = sch.schema([("a", dt.int64), ("c", dt.string)])

    assert s1 > s2
    assert s2 < s1

    assert s1 >= s2
    assert s2 <= s1


def test_empty_schema():
    s1 = sch.Schema({})
    s2 = sch.schema([])

    assert s1 == s2

    for s in [s1, s2]:
        assert len(s.items()) == 0
        assert repr(s) == "ibis.Schema {\n}"


def test_nullable_output():
    s = sch.schema(
        [
            ("foo", "int64"),
            ("bar", dt.int64(nullable=False)),
            ("baz", "boolean"),
        ]
    )

    sch_str = str(s)
    assert "foo  int64" in sch_str
    assert "foo  !int64" not in sch_str
    assert "bar  !int64" in sch_str
    assert "baz  boolean" in sch_str
    assert "baz  !boolean" not in sch_str


@pytest.fixture
def df():
    return pd.DataFrame({"A": pd.Series([1], dtype="int8"), "b": ["x"]})


def test_apply_to_column_rename(df):
    schema = sch.Schema({"a": "int8", "B": "string"})
    expected = df.rename({"A": "a", "b": "B"}, axis=1)
    with pytest.warns(FutureWarning):
        df = schema.apply_to(df.copy())
    tm.assert_frame_equal(df, expected)


def test_apply_to_column_order(df):
    schema = sch.Schema({"a": "int8", "b": "string"})
    expected = df.rename({"A": "a"}, axis=1)
    with pytest.warns(FutureWarning):
        new_df = schema.apply_to(df.copy())
    tm.assert_frame_equal(new_df, expected)


def test_api_accepts_schema_objects():
    s1 = sch.schema(dict(a="int", b="str"))
    s2 = sch.schema(s1)
    assert s1 == s2


def test_schema_mapping_api():
    s = sch.Schema(
        {
            "a": "map<double, string>",
            "b": "array<map<string, array<int32>>>",
            "c": "array<string>",
            "d": "int8",
        }
    )

    assert s["a"] == dt.Map(dt.double, dt.string)
    assert s["b"] == dt.Array(dt.Map(dt.string, dt.Array(dt.int32)))
    assert s["c"] == dt.Array(dt.string)
    assert s["d"] == dt.int8

    assert "a" in s
    assert "e" not in s
    assert len(s) == 4
    assert tuple(s) == s.names
    assert tuple(s.keys()) == s.names
    assert tuple(s.values()) == s.types
    assert tuple(s.items()) == tuple(zip(s.names, s.types))


class BarSchema:
    a: int
    b: str


class FooSchema:
    a: int
    b: str
    c: float
    d: tuple[str]
    e: list[int]
    f: dict[str, int]
    g: BarSchema
    h: list[BarSchema]
    j: dict[str, BarSchema]


foo_schema = sch.Schema(
    {
        "a": "int64",
        "b": "string",
        "c": "float64",
        "d": "array<string>",
        "e": "array<int64>",
        "f": "map<string, int64>",
        "g": "struct<a: int64, b: string>",
        "h": "array<struct<a: int64, b: string>>",
        "j": "map<string, struct<a: int64, b: string>>",
    }
)


def test_schema_from_annotated_class():
    assert sch.schema(FooSchema) == foo_schema


class NamedBar(NamedTuple):
    a: int
    b: str


class NamedFoo(NamedTuple):
    a: int
    b: str
    c: float
    d: tuple[str]
    e: list[int]
    f: dict[str, int]
    g: NamedBar
    h: list[NamedBar]
    j: dict[str, NamedBar]


def test_schema_from_namedtuple():
    assert sch.schema(NamedFoo) == foo_schema


@dataclass
class DataBar:
    a: int
    b: str


@dataclass
class DataFooBase:
    a: int
    b: str
    c: float
    d: tuple[str]


@dataclass
class DataFoo(DataFooBase):
    e: list[int]
    f: dict[str, int]
    g: DataBar
    h: list[DataBar]
    j: dict[str, DataBar]


def test_schema_from_dataclass():
    assert sch.schema(DataFoo) == foo_schema


class PreferenceA:
    a: dt.int64
    b: dt.Array(dt.int64)


class PreferenceB:
    a: dt.int64
    b: dt.Array[dt.int64]


class PreferenceC:
    a: dt.Int64
    b: dt.Array[dt.Int64]


def test_preferences():
    a = sch.schema(PreferenceA)
    b = sch.schema(PreferenceB)
    c = sch.schema(PreferenceC)
    assert a == b == c


class ObjectWithSchema(Annotable):
    schema: sch.Schema


def test_schema_is_coercible():
    s = sch.Schema({"a": dt.int64, "b": dt.Array(dt.int64)})
    assert CoercedTo(sch.Schema).match(PreferenceA, {}) == s

    o = ObjectWithSchema(schema=PreferenceA)
    assert o.schema == s


def test_schema_set_operations():
    a = sch.Schema({"a": dt.string, "b": dt.int64, "c": dt.float64})
    b = sch.Schema({"a": dt.string, "c": dt.float64, "d": dt.boolean, "e": dt.date})
    c = sch.Schema({"i": dt.int64, "j": dt.float64, "k": dt.string})
    d = sch.Schema({"i": dt.int64, "j": dt.float64, "k": dt.string, "l": dt.boolean})

    assert a & b == sch.Schema({"a": dt.string, "c": dt.float64})
    assert a | b == sch.Schema(
        {"a": dt.string, "b": dt.int64, "c": dt.float64, "d": dt.boolean, "e": dt.date}
    )
    assert a - b == sch.Schema({"b": dt.int64})
    assert b - a == sch.Schema({"d": dt.boolean, "e": dt.date})
    assert a ^ b == sch.Schema({"b": dt.int64, "d": dt.boolean, "e": dt.date})

    assert not a.isdisjoint(b)
    assert a.isdisjoint(c)

    assert a <= a
    assert a >= a
    assert not a < a
    assert not a > a
    assert not a <= b
    assert not a >= b
    assert not a >= c
    assert not a <= c
    assert c <= d
    assert c < d
    assert d >= c
    assert d > c


def test_schema_infer_pyarrow_table():
    table = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3]),
            pa.array(["a", "b", "c"]),
            pa.array([True, False, True]),
        ],
        ["a", "b", "c"],
    )
    s = sch.infer(table)
    assert s == sch.Schema({"a": dt.int64, "b": dt.string, "c": dt.boolean})


def test_schema_from_to_pyarrow_schema():
    pyarrow_schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.string()),
            pa.field("c", pa.bool_()),
        ]
    )
    ibis_schema = sch.schema(pyarrow_schema)
    restored_schema = ibis_schema.to_pyarrow()

    assert ibis_schema == sch.Schema({"a": dt.int64, "b": dt.string, "c": dt.boolean})
    assert restored_schema == pyarrow_schema


def test_schema_from_to_numpy_dtypes():
    numpy_dtypes = [
        ("a", np.dtype("int64")),
        ("b", np.dtype("str")),
        ("c", np.dtype("bool")),
    ]
    ibis_schema = sch.Schema.from_numpy(numpy_dtypes)
    assert ibis_schema == sch.Schema({"a": dt.int64, "b": dt.string, "c": dt.boolean})

    restored_dtypes = ibis_schema.to_numpy()
    expected_dtypes = [
        ("a", np.dtype("int64")),
        ("b", np.dtype("object")),
        ("c", np.dtype("bool")),
    ]
    assert restored_dtypes == expected_dtypes


@pytest.mark.parametrize(
    ("from_method", "to_method"),
    [
        pytest.param(
            "from_dask",
            "to_dask",
            marks=pytest.mark.skipif(not has_dask, reason="dask not installed"),
        ),
        pytest.param(
            "from_pandas",
            "to_pandas",
            marks=pytest.mark.skipif(not has_pandas, reason="pandas not installed"),
        ),
    ],
)
def test_schema_from_to_pandas_dask_dtypes(from_method, to_method):
    pandas_schema = pd.Series(
        [
            ("a", np.dtype("int64")),
            ("b", np.dtype("str")),
            ("c", pd.CategoricalDtype(["a", "b", "c"])),
            ("d", pd.DatetimeTZDtype(tz="US/Eastern", unit="ns")),
        ]
    )
    ibis_schema = getattr(sch.Schema, from_method)(pandas_schema)
    assert ibis_schema == sch.schema(pandas_schema)

    expected = sch.Schema(
        {
            "a": dt.int64,
            "b": dt.string,
            "c": dt.string,
            "d": dt.Timestamp(timezone="US/Eastern"),
        }
    )
    assert ibis_schema == expected

    restored_dtypes = getattr(ibis_schema, to_method)()
    expected_dtypes = [
        ("a", np.dtype("int64")),
        ("b", np.dtype("object")),
        ("c", np.dtype("object")),
        ("d", pd.DatetimeTZDtype(tz="US/Eastern", unit="ns")),
    ]
    assert restored_dtypes == expected_dtypes
