from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from decimal import Decimal
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from pytest import param

import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
from ibis.common.exceptions import IntegrityError
from ibis.common.grounds import Annotable


def test_whole_schema():
    schema = {
        'cid': 'int64',
        'mktsegment': 'string',
        'address': '''struct<city: string,
                             street: string,
                             street_number: int32,
                             zip: int16>''',
        'phone_numbers': 'array<string>',
        'orders': '''array<struct<oid: int64,
                                  status: string,
                                  totalprice: decimal(12, 2),
                                  order_date: string,
                                  items: array<struct<iid: int64,
                                                      name: string,
                                                      price: decimal(12, 2),
                                                      discount_perc: decimal(12, 2),
                                                      shipdate: string>>>>
                                                      ''',
        'web_visits': '''map<string, struct<user_agent: string,
                                            client_ip: string,
                                            visit_date: string,
                                            duration_ms: int32>>''',
        'support_calls': '''array<struct<agent_id: int64,
                                         call_date: string,
                                         duration_ms: int64,
                                         issue_resolved: boolean,
                                         agent_comment: string>>''',
    }
    expected = {
        'cid': dt.int64,
        'mktsegment': dt.string,
        'address': dt.Struct(
            {
                'city': dt.string,
                'street': dt.string,
                'street_number': dt.int32,
                'zip': dt.int16,
            }
        ),
        'phone_numbers': dt.Array(dt.string),
        'orders': dt.Array(
            dt.Struct(
                {
                    'oid': dt.int64,
                    'status': dt.string,
                    'totalprice': dt.Decimal(12, 2),
                    'order_date': dt.string,
                    'items': dt.Array(
                        dt.Struct(
                            {
                                'iid': dt.int64,
                                'name': dt.string,
                                'price': dt.Decimal(12, 2),
                                'discount_perc': dt.Decimal(12, 2),
                                'shipdate': dt.string,
                            }
                        )
                    ),
                }
            )
        ),
        'web_visits': dt.Map(
            dt.string,
            dt.Struct(
                {
                    'user_agent': dt.string,
                    'client_ip': dt.string,
                    'visit_date': dt.string,
                    'duration_ms': dt.int32,
                }
            ),
        ),
        'support_calls': dt.Array(
            dt.Struct(
                {
                    'agent_id': dt.int64,
                    'call_date': dt.string,
                    'duration_ms': dt.int64,
                    'issue_resolved': dt.boolean,
                    'agent_comment': dt.string,
                }
            )
        ),
    }
    assert sch.schema(schema) == sch.schema(expected)


def test_schema_from_tuples():
    schema = sch.Schema.from_tuples(
        [
            ('a', 'int64'),
            ('b', 'string'),
            ('c', 'double'),
            ('d', 'boolean'),
        ]
    )
    expected = sch.Schema(
        {'a': dt.int64, 'b': dt.string, 'c': dt.double, 'd': dt.boolean}
    )

    assert schema == expected

    # test that duplicate field names are prohibited
    with pytest.raises(IntegrityError):
        sch.Schema.from_tuples([('a', 'int64'), ('a', 'string')])


def test_schema_subset():
    s1 = sch.schema([('a', dt.int64), ('b', dt.int32), ('c', dt.string)])
    s2 = sch.schema([('a', dt.int64), ('c', dt.string)])

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
            ('foo', 'int64'),
            ('bar', dt.int64(nullable=False)),
            ('baz', 'boolean'),
        ]
    )

    sch_str = str(s)
    assert 'foo  int64' in sch_str
    assert 'foo  !int64' not in sch_str
    assert 'bar  !int64' in sch_str
    assert 'baz  boolean' in sch_str
    assert 'baz  !boolean' not in sch_str


@pytest.fixture
def df():
    return pd.DataFrame({"A": pd.Series([1], dtype="int8"), "b": ["x"]})


def test_apply_to_column_rename(df):
    schema = sch.Schema({"a": "int8", "B": "string"})
    expected = df.rename({"A": "a", "b": "B"}, axis=1)
    tm.assert_frame_equal(schema.apply_to(df.copy()), expected)


def test_apply_to_column_order(df):
    schema = sch.Schema({"a": "int8", "b": "string"})
    expected = df.rename({"A": "a"}, axis=1)
    new_df = schema.apply_to(df.copy())
    tm.assert_frame_equal(new_df, expected)


def test_api_accepts_schema_objects():
    s1 = sch.schema(dict(a="int", b="str"))
    s2 = sch.schema(s1)
    assert s1 == s2


def test_schema_mapping_api():
    s = sch.Schema(
        {
            'a': 'map<double, string>',
            'b': 'array<map<string, array<int32>>>',
            'c': 'array<string>',
            'd': 'int8',
        }
    )

    assert s['a'] == dt.Map(dt.double, dt.string)
    assert s['b'] == dt.Array(dt.Map(dt.string, dt.Array(dt.int32)))
    assert s['c'] == dt.Array(dt.string)
    assert s['d'] == dt.int8

    assert 'a' in s
    assert 'e' not in s
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
    d: Tuple[str]  # noqa: UP006
    e: List[int]  # noqa: UP006
    f: Dict[str, int]  # noqa: UP006
    g: BarSchema
    h: List[BarSchema]  # noqa: UP006
    j: Dict[str, BarSchema]  # noqa: UP006


foo_schema = sch.Schema(
    {
        'a': 'int64',
        'b': 'string',
        'c': 'float64',
        'd': 'array<string>',
        'e': 'array<int64>',
        'f': 'map<string, int64>',
        'g': 'struct<a: int64, b: string>',
        'h': 'array<struct<a: int64, b: string>>',
        'j': 'map<string, struct<a: int64, b: string>>',
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
    d: Tuple[str]  # noqa: UP006
    e: List[int]  # noqa: UP006
    f: Dict[str, int]  # noqa: UP006
    g: NamedBar
    h: List[NamedBar]  # noqa: UP006
    j: Dict[str, NamedBar]  # noqa: UP006


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
    d: Tuple[str]  # noqa: UP006


@dataclass
class DataFoo(DataFooBase):
    e: List[int]  # noqa: UP006
    f: Dict[str, int]  # noqa: UP006
    g: DataBar
    h: List[DataBar]  # noqa: UP006
    j: Dict[str, DataBar]  # noqa: UP006


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
    s = sch.Schema({'a': dt.int64, 'b': dt.Array(dt.int64)})
    assert rlz.coerced_to(sch.Schema, PreferenceA) == s

    o = ObjectWithSchema(schema=PreferenceA)
    assert o.schema == s


def test_schema_set_operations():
    a = sch.Schema({'a': dt.string, 'b': dt.int64, 'c': dt.float64})
    b = sch.Schema({'a': dt.string, 'c': dt.float64, 'd': dt.boolean, 'e': dt.date})
    c = sch.Schema({'i': dt.int64, 'j': dt.float64, 'k': dt.string})
    d = sch.Schema({'i': dt.int64, 'j': dt.float64, 'k': dt.string, 'l': dt.boolean})

    assert a & b == sch.Schema({'a': dt.string, 'c': dt.float64})
    assert a | b == sch.Schema(
        {'a': dt.string, 'b': dt.int64, 'c': dt.float64, 'd': dt.boolean, 'e': dt.date}
    )
    assert a - b == sch.Schema({'b': dt.int64})
    assert b - a == sch.Schema({'d': dt.boolean, 'e': dt.date})
    assert a ^ b == sch.Schema({'b': dt.int64, 'd': dt.boolean, 'e': dt.date})

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
            pa.array(['a', 'b', 'c']),
            pa.array([True, False, True]),
        ],
        ['a', 'b', 'c'],
    )
    s = sch.infer(table)
    assert s == sch.Schema({'a': dt.int64, 'b': dt.string, 'c': dt.boolean})


def test_schema_from_pyarrow_schema():
    schema = pa.schema(
        [
            pa.field('a', pa.int64()),
            pa.field('b', pa.string()),
            pa.field('c', pa.bool_()),
        ]
    )
    s = sch.schema(schema)
    assert s == sch.Schema({'a': dt.int64, 'b': dt.string, 'c': dt.boolean})


@pytest.mark.parametrize(
    ('col_data', 'schema_type'),
    [
        param([True, False, False], 'bool', id="bool"),
        param(np.array([-3, 9, 17], dtype='int8'), 'int8', id="int8"),
        param(np.array([-5, 0, 12], dtype='int16'), 'int16', id="int16"),
        param(np.array([-12, 3, 25000], dtype='int32'), 'int32', id="int32"),
        param(np.array([102, 67228734, -0], dtype='int64'), 'int64', id="int64"),
        param(np.array([45e-3, -0.4, 99.0], dtype='float32'), 'float32', id="float64"),
        param(np.array([45e-3, -0.4, 99.0], dtype='float64'), 'float64', id="float32"),
        param(
            np.array([-3e43, 43.0, 10000000.0], dtype='float64'), 'double', id="double"
        ),
        param(np.array([3, 0, 16], dtype='uint8'), 'uint8', id="uint8"),
        param(np.array([5569, 1, 33], dtype='uint16'), 'uint16', id="uint8"),
        param(np.array([100, 0, 6], dtype='uint32'), 'uint32', id="uint32"),
        param(np.array([666, 2, 3], dtype='uint64'), 'uint64', id="uint64"),
        param(
            [
                pd.Timestamp('2010-11-01 00:01:00'),
                pd.Timestamp('2010-11-01 00:02:00.1000'),
                pd.Timestamp('2010-11-01 00:03:00.300000'),
            ],
            'timestamp',
            id="timestamp",
        ),
        param(
            [
                pd.Timedelta('1 days'),
                pd.Timedelta('-1 days 2 min 3us'),
                pd.Timedelta('-2 days +23:57:59.999997'),
            ],
            "interval('ns')",
            id="interval_ns",
        ),
        param(['foo', 'bar', 'hello'], "string", id="string_list"),
        param(
            pd.Series(['a', 'b', 'c', 'a']).astype('category'),
            dt.String(),
            id="string_series",
        ),
        param(pd.Series([b'1', b'2', b'3']), dt.binary, id="string_binary"),
        # mixed-integer
        param(pd.Series([1, 2, '3']), dt.unknown, id="mixed_integer"),
        # mixed-integer-float
        param(pd.Series([1, 2, 3.0]), dt.float64, id="mixed_integer_float"),
        param(
            pd.Series([Decimal('1.0'), Decimal('2.0'), Decimal('3.0')]),
            dt.Decimal(2, 1),
            id="decimal",
        ),
        # complex
        param(
            pd.Series([1 + 1j, 1 + 2j, 1 + 3j], dtype=object), dt.unknown, id="complex"
        ),
        param(
            pd.Series(
                [
                    pd.to_datetime('2010-11-01'),
                    pd.to_datetime('2010-11-02'),
                    pd.to_datetime('2010-11-03'),
                ]
            ),
            dt.timestamp,
            id="timestamp_to_datetime",
        ),
        param(pd.Series([time(1), time(2), time(3)]), dt.time, id="time"),
        param(
            pd.Series(
                [
                    pd.Period('2011-01'),
                    pd.Period('2011-02'),
                    pd.Period('2011-03'),
                ],
                dtype=object,
            ),
            dt.unknown,
            id="period",
        ),
        # mixed
        param(pd.Series([b'1', '2', 3.0]), dt.unknown, id="mixed"),
        # empty
        param(pd.Series([], dtype='object'), dt.null, id="empty_null"),
        param(pd.Series([], dtype="string"), dt.string, id="empty_string"),
        # array
        param(pd.Series([[1], [], None]), dt.Array(dt.int64), id="array_int64_first"),
        param(pd.Series([[], [1], None]), dt.Array(dt.int64), id="array_int64_second"),
    ],
)
def test_infer_pandas_dataframe_schema(col_data, schema_type):
    df = pd.DataFrame({'col': col_data})

    inferred = sch.infer(df)
    expected = sch.schema([('col', schema_type)])
    assert inferred == expected
