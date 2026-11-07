from __future__ import annotations

import pytest
from pyflink.common.types import Row

import ibis
from ibis.backends.tests.errors import Py4JJavaError
from ibis.conftest import IS_PYFLINK_1X


def struct(*fields):
    # pyflink 1.x hands a nested ROW back as a Row holding a single list of
    # its fields rather than one holding the fields themselves
    return Row(list(fields)) if IS_PYFLINK_1X else Row(*fields)


@pytest.mark.parametrize(
    ("data", "schema", "expected"),
    [
        pytest.param(
            {"value": [{"a": 1}, {"a": 2}]},
            {"value": "!struct<a: !int>"},
            [Row(struct(1)), Row(struct(2))],
            id="simple_named_struct",
        ),
        pytest.param(
            {"value": [[{"a": 1}, {"a": 2}], [{"a": 3}, {"a": 4}]]},
            {"value": "!array<!struct<a: !int>>"},
            [Row([struct(1), struct(2)]), Row([struct(3), struct(4)])],
            id="single_field_named_struct_array",
        ),
        pytest.param(
            {"value": [[{"a": 1, "b": 2}, {"a": 2, "b": 2}]]},
            {"value": "!array<!struct<a: !int, b: !int>>"},
            [Row([struct(1, 2), struct(2, 2)])],
            id="named_struct_array",
        ),
    ],
)
def test_create_memtable(con, data, schema, expected):
    t = ibis.memtable(data, schema=ibis.schema(schema))
    # cannot use con.execute(t) directly because of some behavioral discrepancy between
    # `TableEnvironment.execute_sql()` and `TableEnvironment.sql_query()`; this doesn't
    # seem to be an issue if we don't execute memtable directly
    result = list(con.raw_sql(con.compile(t)).collect())
    for element in expected:
        assert element in result


@pytest.mark.notyet(
    ["flink"],
    condition=IS_PYFLINK_1X,
    raises=Py4JJavaError,
    reason="cannot create an ARRAY of named STRUCTs directly from the ARRAY[] constructor; https://issues.apache.org/jira/browse/FLINK-34898",
)
def test_create_named_struct_array_with_array_constructor(con):
    con.raw_sql("SELECT ARRAY[cast(ROW(1) as ROW<a INT>)];")
