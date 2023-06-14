import pytest
from pytest import param

import ibis
from ibis.backends.flink.compiler.core import translate


@pytest.fixture
def schema():
    return [
        ('a', 'int8'),
        ('b', 'int16'),
        ('c', 'int32'),
        ('d', 'int64'),
        ('e', 'float32'),
        ('f', 'float64'),
        ('g', 'string'),
        ('h', 'boolean'),
        ('i', 'timestamp'),
        ('j', 'date'),
        ('k', 'time'),
    ]


@pytest.fixture
def table(schema):
    return ibis.table(schema, name='table')


def test_translate_sum(snapshot, table):
    expr = table.a.sum()
    result = translate(expr.as_table().op())
    snapshot.assert_match(str(result), "out.sql")


def test_translate_count_star(snapshot, table):
    expr = table.group_by(table.i).size()
    result = translate(expr.as_table().op())
    snapshot.assert_match(str(result), "out.sql")


@pytest.mark.parametrize(
    "unit",
    [
        param("ms", id="timestamp_ms"),
        param("s", id="timestamp_s"),
    ],
)
def test_translate_timestamp_from_unix(snapshot, table, unit):
    expr = table.d.to_timestamp(unit=unit)
    result = translate(expr.as_table().op())
    snapshot.assert_match(result, "out.sql")
