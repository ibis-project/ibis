from __future__ import annotations

import builtins
import contextlib
import importlib
import inspect
import json
import re
import string
import subprocess
import sys
from operator import itemgetter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import rich.console
import toolz
from packaging.version import parse as vparse
from pytest import mark, param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.conftest import ALL_BACKENDS
from ibis.backends.tests.errors import (
    ExaQueryError,
    ImpalaHiveServer2Error,
    OracleDatabaseError,
    PsycoPg2InternalError,
    PsycoPg2UndefinedObject,
    PyODBCProgrammingError,
    SnowflakeProgrammingError,
    TrinoUserError,
)
from ibis.util import gen_name

if TYPE_CHECKING:
    from ibis.backends import BaseBackend


@pytest.fixture
def new_schema():
    return ibis.schema([("a", "string"), ("b", "bool"), ("c", "int32")])


def _create_temp_table_with_schema(backend, con, temp_table_name, schema, data=None):
    if con.name == "druid":
        pytest.xfail("druid doesn't implement create_table")
    elif con.name == "flink":
        pytest.xfail(
            "flink doesn't implement create_table from schema without additional arguments"
        )
    temporary = con.create_table(temp_table_name, schema=schema)
    assert temporary.to_pandas().empty

    if data is not None and isinstance(data, pd.DataFrame):
        assert not data.empty
        tmp = con.create_table(temp_table_name, data, overwrite=True)
        result = tmp.to_pandas()
        assert len(result) == len(data.index)
        backend.assert_frame_equal(
            result.sort_values(result.columns[0]).reset_index(drop=True),
            data.sort_values(result.columns[0]).reset_index(drop=True),
        )
        return tmp

    return temporary


@pytest.mark.parametrize(
    "func", [toolz.identity, pa.Table.from_pandas], ids=["dataframe", "pyarrow_table"]
)
@pytest.mark.parametrize(
    "sch",
    [
        None,
        ibis.schema(
            dict(
                first_name="string",
                last_name="string",
                department_name="string",
                salary="float64",
            )
        ),
    ],
    ids=["no_schema", "schema"],
)
@pytest.mark.notimpl(["druid"])
@pytest.mark.notimpl(
    ["flink"],
    reason="Flink backend supports creating only TEMPORARY VIEW for in-memory data.",
)
def test_create_table(backend, con, temp_table, func, sch):
    df = pd.DataFrame(
        {
            "first_name": ["A", "B", "C"],
            "last_name": ["D", "E", "F"],
            "department_name": ["AA", "BB", "CC"],
            "salary": [100.0, 200.0, 300.0],
        }
    )

    con.create_table(temp_table, func(df), schema=sch)
    result = (
        con.table(temp_table).execute().sort_values("first_name").reset_index(drop=True)
    )

    backend.assert_frame_equal(df, result)


@pytest.mark.parametrize(
    "temp, overwrite",
    [
        param(
            True,
            True,
            id="temp overwrite",
            marks=[
                pytest.mark.notyet(["clickhouse"], reason="Can't specify both"),
                pytest.mark.notyet(
                    ["pyspark", "trino", "exasol", "risingwave"],
                    reason="No support for temp tables",
                ),
                pytest.mark.broken(["mssql"], reason="Incorrect temp table syntax"),
                pytest.mark.broken(
                    ["bigquery"],
                    reason="tables created with temp=True cause a 404 on retrieval",
                ),
            ],
        ),
        param(
            False,
            True,
            id="no temp, overwrite",
            marks=pytest.mark.notyet(["flink", "polars"]),
        ),
        param(
            True,
            False,
            id="temp, no overwrite",
            marks=[
                pytest.mark.notyet(
                    ["pyspark", "trino", "exasol", "risingwave"],
                    reason="No support for temp tables",
                ),
                pytest.mark.broken(["mssql"], reason="Incorrect temp table syntax"),
                pytest.mark.broken(
                    ["bigquery"],
                    reason="tables created with temp=True cause a 404 on retrieval",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(["druid", "impala"])
def test_create_table_overwrite_temp(backend, con, temp_table, temp, overwrite):
    df = pd.DataFrame(
        {
            "first_name": ["A", "B", "C"],
            "last_name": ["D", "E", "F"],
            "department_name": ["AA", "BB", "CC"],
            "salary": [100.0, 200.0, 300.0],
        }
    )

    con.create_table(temp_table, df, temp=temp, overwrite=overwrite)
    if overwrite:
        con.create_table(temp_table, df, temp=temp, overwrite=overwrite)
    result = (
        con.table(temp_table).execute().sort_values("first_name").reset_index(drop=True)
    )

    backend.assert_frame_equal(df, result)


@pytest.mark.parametrize(
    "lamduh",
    [(lambda df: df), (lambda df: pa.Table.from_pandas(df))],
    ids=["dataframe", "pyarrow table"],
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(["flink"], raises=com.IbisError)
def test_load_data(backend, con, temp_table, lamduh):
    sch = ibis.schema(
        [
            ("first_name", "string"),
            ("last_name", "string"),
            ("department_name", "string"),
            ("salary", "float64"),
        ]
    )

    df = pd.DataFrame(
        {
            "first_name": ["A", "B", "C"],
            "last_name": ["D", "E", "F"],
            "department_name": ["AA", "BB", "CC"],
            "salary": [100.0, 200.0, 300.0],
        }
    )

    obj = lamduh(df)
    con.create_table(temp_table, obj, schema=sch, overwrite=True)
    result = (
        con.table(temp_table).execute().sort_values("first_name").reset_index(drop=True)
    )

    backend.assert_frame_equal(df, result)


@mark.parametrize(
    ("expr_fn", "expected"),
    [
        param(lambda t: t.string_col, [("string_col", dt.String)], id="column"),
        param(
            lambda t: t[t.string_col, t.bigint_col],
            [("string_col", dt.String), ("bigint_col", dt.Int64)],
            id="table",
        ),
    ],
)
def test_query_schema(ddl_backend, expr_fn, expected):
    expr = expr_fn(ddl_backend.functional_alltypes)

    # we might need a public API for it
    schema = expr.as_table().schema()

    # clickhouse columns has been defined as non-nullable
    # whereas other backends don't support non-nullable columns yet
    expected = ibis.schema(
        [(name, dtype(nullable=schema[name].nullable)) for name, dtype in expected]
    )
    assert schema.equals(expected)


_LIMIT = {
    "oracle": "FETCH FIRST 10 ROWS ONLY",
}


@pytest.mark.notimpl(["datafusion", "mssql"])
@pytest.mark.never(["dask", "pandas"], reason="dask and pandas do not support SQL")
def test_sql(backend, con):
    # execute the expression using SQL query
    table = backend.format_table("functional_alltypes")
    limit = _LIMIT.get(backend.name(), "LIMIT 10")
    expr = con.sql(f"SELECT * FROM {table} {limit}")
    result = expr.execute()
    assert len(result) == 10


backend_type_mapping = {
    # backends only implement int64
    "bigquery": {
        dt.int32: dt.int64,
    },
    "snowflake": {
        dt.int32: dt.int64,
    },
}


@mark.notimpl(["druid"])
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
def test_create_table_from_schema(con, new_schema, temp_table):
    new_table = con.create_table(temp_table, schema=new_schema)
    backend_mapping = backend_type_mapping.get(con.name, {})

    result = ibis.schema(
        {
            column_name: backend_mapping.get(
                new_schema[column_name], new_schema[column_name]
            )
            for column_name in new_table.schema().keys()
        }
    )
    assert result == new_table.schema()


@mark.broken(
    ["oracle"],
    reason="oracle temp tables aren't cleaned up on reconnect -- they need to "
    "be switched from using atexit to weakref.finalize",
)
@mark.notimpl(["trino", "druid"], reason="doesn't implement temporary tables")
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
    raises=PyODBCProgrammingError,
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.notimpl(
    ["impala", "pyspark"],
    reason="temporary tables not implemented",
    raises=NotImplementedError,
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="truncate not supported upstream",
)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
def test_create_temporary_table_from_schema(con_no_data, new_schema):
    temp_table = gen_name(f"test_{con_no_data.name}_tmp")
    table = con_no_data.create_table(temp_table, schema=new_schema, temp=True)

    # verify table exist in the current session
    backend_mapping = backend_type_mapping.get(con_no_data.name, dict())
    for column_name, column_type in table.schema().items():
        assert (
            backend_mapping.get(new_schema[column_name], new_schema[column_name])
            == column_type
        )

    con_no_data.reconnect()
    # verify table no longer exist after reconnect
    assert temp_table not in con_no_data.tables.keys()


@mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "datafusion",
        "druid",
        "duckdb",
        "exasol",
        "mssql",
        "mysql",
        "oracle",
        "pandas",
        "polars",
        "postgres",
        "risingwave",
        "snowflake",
        "sqlite",
        "trino",
    ]
)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
def test_rename_table(con, temp_table, temp_table_orig):
    schema = ibis.schema({"a": "string", "b": "bool", "c": "int32"})
    con.create_table(temp_table_orig, schema=schema)
    con.rename_table(temp_table_orig, temp_table)
    new = con.table(temp_table)
    assert new.schema().equals(schema)
    assert temp_table_orig not in con.list_tables()


@mark.notimpl(["polars", "druid"])
@mark.never(["impala", "pyspark"], reason="No non-nullable datatypes")
@mark.notyet(
    ["trino"], reason="trino doesn't support NOT NULL in its in-memory catalog"
)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason='Feature is not yet implemented: column constraints "NOT NULL"',
)
def test_nullable_input_output(con, temp_table):
    sch = ibis.schema(
        [("foo", "int64"), ("bar", dt.int64(nullable=False)), ("baz", "boolean")]
    )
    t = con.create_table(temp_table, schema=sch)

    assert t.schema().types[0].nullable
    assert not t.schema().types[1].nullable
    assert t.schema().types[2].nullable


@mark.notimpl(["druid", "polars"])
def test_create_drop_view(ddl_con, temp_view):
    # setup
    table_name = "functional_alltypes"
    tables = ddl_con.list_tables()

    if table_name in tables or (table_name := table_name.upper()) in tables:
        expr = ddl_con.table(table_name)
    else:
        raise ValueError(f"table `{table_name}` does not exist")

    expr = expr.limit(1)

    # create a new view
    ddl_con.create_view(temp_view, expr)
    # check if the view was created
    assert temp_view in ddl_con.list_tables()

    t_expr = ddl_con.table(table_name)
    v_expr = ddl_con.table(temp_view)
    # check if the view and the table has the same fields
    assert set(t_expr.schema().names) == set(v_expr.schema().names)


@pytest.fixture
def employee_empty_temp_table(backend, con, test_employee_schema):
    temp_table_name = gen_name("temp_employee_empty_table")
    _create_temp_table_with_schema(backend, con, temp_table_name, test_employee_schema)
    yield temp_table_name
    con.drop_table(temp_table_name, force=True)


@pytest.fixture
def employee_data_1_temp_table(
    backend, con, test_employee_schema, test_employee_data_1
):
    temp_table_name = gen_name("temp_employee_data_1")
    _create_temp_table_with_schema(
        backend, con, temp_table_name, test_employee_schema, data=test_employee_data_1
    )
    assert temp_table_name in con.list_tables()
    yield temp_table_name
    con.drop_table(temp_table_name, force=True)


@pytest.fixture
def employee_data_2_temp_table(
    backend, con, test_employee_schema, test_employee_data_2
):
    temp_table_name = gen_name("temp_employee_data_2")
    _create_temp_table_with_schema(
        backend, con, temp_table_name, test_employee_schema, data=test_employee_data_2
    )
    yield temp_table_name
    con.drop_table(temp_table_name, force=True)


@pytest.mark.notimpl(
    ["polars", "pandas", "dask"], reason="`insert` method not implemented"
)
def test_insert_no_overwrite_from_dataframe(
    backend, con, test_employee_data_2, employee_empty_temp_table
):
    temporary = con.table(employee_empty_temp_table)
    con.insert(employee_empty_temp_table, obj=test_employee_data_2, overwrite=False)
    result = temporary.execute()
    assert len(result) == 3
    backend.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        test_employee_data_2.sort_values("first_name").reset_index(drop=True),
    )


@pytest.mark.notimpl(
    ["polars", "pandas", "dask"], reason="`insert` method not implemented"
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="truncate not supported upstream",
)
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(
    ["trino"], raises=TrinoUserError, reason="requires a non-memory connector"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
def test_insert_overwrite_from_dataframe(
    backend, con, employee_data_1_temp_table, test_employee_data_2
):
    temporary = con.table(employee_data_1_temp_table)

    con.insert(employee_data_1_temp_table, obj=test_employee_data_2, overwrite=True)
    result = temporary.execute()
    assert len(result) == 3
    backend.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        test_employee_data_2.sort_values("first_name").reset_index(drop=True),
    )


@pytest.mark.notimpl(
    ["polars", "pandas", "dask"], reason="`insert` method not implemented"
)
def test_insert_no_overwrite_from_expr(
    backend, con, employee_empty_temp_table, employee_data_2_temp_table
):
    temporary = con.table(employee_empty_temp_table)
    from_table = con.table(employee_data_2_temp_table)

    con.insert(employee_empty_temp_table, obj=from_table, overwrite=False)
    result = temporary.execute()
    assert len(result) == 3
    backend.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        from_table.execute().sort_values("first_name").reset_index(drop=True),
    )


@pytest.mark.notimpl(
    ["polars", "pandas", "dask"], reason="`insert` method not implemented"
)
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(
    ["trino"],
    raises=TrinoUserError,
    reason="requires a non-memory connector for truncation",
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="truncate not supported upstream",
)
def test_insert_overwrite_from_expr(
    backend, con, employee_data_1_temp_table, employee_data_2_temp_table
):
    temporary = con.table(employee_data_1_temp_table)
    from_table = con.table(employee_data_2_temp_table)

    con.insert(employee_data_1_temp_table, obj=from_table, overwrite=True)
    result = temporary.execute()
    assert len(result) == 3
    backend.assert_frame_equal(
        result.sort_values("first_name").reset_index(drop=True),
        from_table.execute().sort_values("first_name").reset_index(drop=True),
    )


@pytest.mark.notyet(
    ["trino"], reason="memory connector doesn't allow writing to tables"
)
@pytest.mark.notimpl(
    ["polars", "pandas", "dask"], reason="`insert` method not implemented"
)
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="truncate not supported upstream",
)
def test_insert_overwrite_from_list(con, employee_data_1_temp_table):
    def _emp(a, b, c, d):
        return dict(first_name=a, last_name=b, department_name=c, salary=d)

    con.insert(
        employee_data_1_temp_table,
        [
            _emp("Adam", "Smith", "Accounting", 50000.0),
            _emp("Mohammed", "Ali", "Boxing", 150000),
            _emp("María", "Gonzalez", "Engineering", 100000.0),
        ],
        overwrite=True,
    )

    assert len(con.table(employee_data_1_temp_table).execute()) == 3


@pytest.mark.notimpl(
    ["polars", "dask", "pandas"],
    raises=AttributeError,
    reason="`insert` method not implemented",
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
def test_insert_from_memtable(con, temp_table):
    df = pd.DataFrame({"x": range(3)})
    table_name = temp_table
    mt = ibis.memtable(df)
    con.create_table(table_name, schema=mt.schema())
    con.insert(table_name, mt)
    con.insert(table_name, mt)

    table = con.tables[table_name]
    assert len(table.execute()) == 6
    assert con.tables[table_name].schema() == ibis.schema({"x": "int64"})


@pytest.mark.notyet(
    ["bigquery", "oracle", "dask", "exasol", "polars", "pandas", "druid"],
    raises=AttributeError,
    reason="doesn't support the common notion of a database",
)
def test_list_databases(con):
    # Every backend has its own databases
    test_databases = {
        "clickhouse": {"system", "default", "ibis_testing"},
        "datafusion": {"datafusion"},
        "duckdb": {"memory"},
        "exasol": set(),
        "impala": set(),
        "mssql": {"ibis_testing"},
        "mysql": {"ibis_testing", "information_schema"},
        "oracle": set(),
        "postgres": {"postgres", "ibis_testing"},
        "risingwave": {"dev"},
        "snowflake": {"IBIS_TESTING"},
        "pyspark": set(),
        "sqlite": {"main"},
        "trino": {"memory"},
        "flink": set(),
    }
    result = set(con.list_databases())
    assert test_databases[con.name] <= result


@pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError)
@pytest.mark.notyet(["pyspark"], raises=com.IbisTypeError)
@pytest.mark.notyet(["bigquery"], raises=com.UnsupportedBackendType)
@pytest.mark.notyet(
    ["postgres"], raises=PsycoPg2UndefinedObject, reason="no unsigned int types"
)
@pytest.mark.notyet(
    ["oracle"], raises=OracleDatabaseError, reason="no unsigned int types"
)
@pytest.mark.notyet(["exasol"], raises=ExaQueryError, reason="no unsigned int types")
@pytest.mark.notyet(["datafusion"], raises=Exception, reason="no unsigned int types")
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError)
@pytest.mark.notyet(["impala"], raises=ImpalaHiveServer2Error)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="unsigned integers are not supported",
)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
def test_unsigned_integer_type(con, temp_table):
    con.create_table(
        temp_table,
        schema=ibis.schema(dict(a="uint8", b="uint16", c="uint32", d="uint64")),
        overwrite=True,
    )
    assert temp_table in con.list_tables()


@pytest.mark.backend
@pytest.mark.parametrize(
    "url",
    [
        param(
            "clickhouse://default@localhost:8123/default",
            marks=mark.clickhouse,
            id="clickhouse",
        ),
        param(
            "dask://",
            marks=[mark.dask, mark.xfail(raises=NotImplementedError)],
            id="dask",
        ),
        param(
            "datafusion://",
            marks=[mark.datafusion, mark.xfail(raises=NotImplementedError)],
            id="datafusion",
        ),
        param(
            "impala://localhost:21050/default",
            marks=mark.impala,
            id="impala",
        ),
        param(
            "mysql://ibis:ibis@localhost:3306",
            marks=mark.mysql,
            id="mysql",
        ),
        param(
            "pandas://",
            marks=[mark.pandas, mark.xfail(raises=NotImplementedError)],
            id="pandas",
        ),
        param(
            "postgres://postgres:postgres@localhost:5432",
            marks=mark.postgres,
            id="postgres",
        ),
        param(
            "postgresql://postgres:postgres@localhost:5432/postgres",
            marks=mark.postgres,
            id="postgresql",
        ),
        param(
            "pyspark://?spark.app.name=test-pyspark",
            marks=[
                mark.pyspark,
                pytest.mark.skipif(
                    sys.version_info >= (3, 11)
                    or vparse(pd.__version__) >= vparse("2")
                    or vparse(np.__version__) >= vparse("1.24"),
                    reason="passes on 3.11, but no other pyspark tests do",
                ),
            ],
            id="pyspark",
        ),
        param(
            "pyspark://my-warehouse-dir?spark.app.name=test-pyspark",
            marks=[
                mark.pyspark,
                pytest.mark.skipif(
                    sys.version_info >= (3, 11)
                    or vparse(pd.__version__) >= vparse("2")
                    or vparse(np.__version__) >= vparse("1.24"),
                    reason="passes on 3.11, but no other pyspark tests do",
                ),
            ],
            id="pyspark_with_warehouse",
        ),
        param(
            "pyspark://my-warehouse-dir",
            marks=[
                mark.pyspark,
                pytest.mark.skipif(
                    sys.version_info >= (3, 11)
                    or vparse(pd.__version__) >= vparse("2")
                    or vparse(np.__version__) >= vparse("1.24"),
                    reason="passes on 3.11, but no other pyspark tests do",
                ),
            ],
            id="pyspark_with_warehouse_no_params",
        ),
    ],
)
def test_connect_url(url):
    con = ibis.connect(url)
    one = ibis.literal(1)
    assert con.execute(one) == 1


@pytest.mark.parametrize(
    ("arg", "lambda_", "expected"),
    [
        param(
            [(1, 2.0, "3")],
            lambda arg: ibis.memtable(arg, columns=list("abc")),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")),
            id="simple",
        ),
        param(
            [(1, 2.0, "3")],
            lambda arg: ibis.memtable(arg),
            pd.DataFrame([(1, 2.0, "3")], columns=["col0", "col1", "col2"]),
            id="simple_auto_named",
        ),
        param(
            [(1, 2.0, "3")],
            lambda arg: ibis.memtable(
                arg,
                schema=ibis.schema(dict(a="int8", b="float32", c="string")),
            ),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")).astype(
                {"a": "int8", "b": "float32"}
            ),
            id="simple_schema",
        ),
        param(
            pd.DataFrame({"a": [1], "b": [2.0], "c": ["3"]}).astype(
                {"a": "int8", "b": "float32"}
            ),
            lambda arg: ibis.memtable(arg),
            pd.DataFrame([(1, 2.0, "3")], columns=list("abc")).astype(
                {"a": "int8", "b": "float32"}
            ),
            id="dataframe",
        ),
        param(
            [dict(a=1), dict(a=2)],
            lambda arg: ibis.memtable(arg),
            pd.DataFrame({"a": [1, 2]}),
            id="list_of_dicts",
        ),
    ],
)
def test_in_memory_table(backend, con, arg, lambda_, expected, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    expr = lambda_(arg)
    result = con.execute(expr.order_by(expr.columns[0]))
    backend.assert_frame_equal(result, expected)


def test_filter_memory_table(backend, con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    t = ibis.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.filter(t.x > 1).order_by("x")
    expected = pd.DataFrame({"x": [3, 5], "y": [4, 6]})
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)


def test_agg_memory_table(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    t = ibis.memtable([(1, 2), (3, 4), (5, 6)], columns=["x", "y"])
    expr = t.x.count()
    result = con.execute(expr)
    assert result == 3


def test_self_join_memory_table(backend, con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    t = ibis.memtable({"x": [1, 2], "y": [2, 1], "z": ["a", "b"]})
    t_view = t.view()
    expr = t.join(t_view, t.x == t_view.y).select("x", "y", "z", "z_right")
    result = con.execute(expr).sort_values("x").reset_index(drop=True)
    expected = pd.DataFrame(
        {"x": [1, 2], "y": [2, 1], "z": ["a", "b"], "z_right": ["b", "a"]}
    )
    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("arg", "func"),
    [
        ([("a", 1.0)], lambda arg: ibis.memtable(arg, columns=["a", "b"])),
        (pd.DataFrame([("a", 1.0)], columns=["a", "b"]), ibis.memtable),
    ],
    ids=["python", "pandas"],
)
@pytest.mark.notimpl(["druid"])
@pytest.mark.notimpl(
    ["flink"],
    reason="Flink backend supports creating only TEMPORARY VIEW for in-memory data.",
)
def test_create_from_in_memory_table(con, temp_table, arg, func, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    t = func(arg)
    con.create_table(temp_table, t)
    assert temp_table in con.list_tables()


def test_default_backend_option(con, monkeypatch):
    # verify that there's nothing already set
    assert ibis.options.default_backend is None

    monkeypatch.setattr(ibis.options, "default_backend", con)

    backend = ibis.config._default_backend()
    assert backend.name == con.name


# backend is used to ensure that this test runs in CI in the setting
# where only the dependencies for a given backend are installed
@pytest.mark.usefixtures("backend")
def test_default_backend_no_duckdb():
    script = """\
import sys
sys.modules["duckdb"] = None

import ibis

t = ibis.memtable([{'a': 1}, {'a': 2}, {'a': 3}])
t.execute()"""

    args = [sys.executable, "-c", script]
    with pytest.raises(subprocess.CalledProcessError) as e:
        subprocess.check_output(args, stderr=subprocess.STDOUT, universal_newlines=True)
    assert (
        re.search(
            "You have used a function that relies on the default backend",
            e.value.output,
        )
        is not None
    )


@pytest.mark.usefixtures("backend")
def test_default_backend_no_duckdb_read_parquet():
    script = """\
import sys
sys.modules["duckdb"] = None

import ibis
ibis.read_parquet("foo.parquet")"""

    args = [sys.executable, "-c", script]
    with pytest.raises(subprocess.CalledProcessError) as e:
        subprocess.check_output(args, stderr=subprocess.STDOUT, universal_newlines=True)
    assert (
        re.search(
            "You have used a function that relies on the default backend",
            e.value.output,
        )
        is not None
    )


@pytest.mark.parametrize("dtype", [None, "f8"])
def test_dunder_array_table(alltypes, dtype):
    expr = alltypes.group_by("string_col").int_col.sum().order_by("string_col")
    result = np.asarray(expr, dtype=dtype)
    expected = np.asarray(expr.execute(), dtype=dtype)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [None, "f8"])
def test_dunder_array_column(alltypes, dtype):
    from ibis import _

    expr = alltypes.group_by("string_col").agg(int_col=_.int_col.sum()).int_col
    result = np.sort(np.asarray(expr, dtype=dtype))
    expected = np.sort(np.asarray(expr.execute(), dtype=dtype))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("interactive", [True, False])
def test_repr(alltypes, interactive, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", interactive)

    expr = alltypes.select("date_string_col")

    s = repr(expr)
    # no control characters
    assert all(c.isprintable() or c in "\n\r\t" for c in s)
    if interactive:
        assert "/" in s
    else:
        assert "/" not in s


@pytest.mark.parametrize("show_types", [True, False])
def test_interactive_repr_show_types(alltypes, show_types, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)
    monkeypatch.setattr(ibis.options.repr.interactive, "show_types", show_types)

    expr = alltypes.select("id")
    s = repr(expr)
    if show_types:
        assert "int" in s
    else:
        assert "int" not in s


@pytest.mark.parametrize("is_jupyter", [True, False])
def test_interactive_repr_max_columns(alltypes, is_jupyter, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)

    cols = {f"c_{i}": ibis._.id + i for i in range(50)}
    expr = alltypes.mutate(**cols).select(*cols)

    console = rich.console.Console(force_jupyter=is_jupyter, width=80)
    options = console.options.copy()

    # max_columns = 0
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_0 " in text
    if is_jupyter:
        # All columns are written
        assert " c_49 " in text
    else:
        # width calculations truncates well before 20 columns
        assert " c_19 " not in text

    # max_columns = 3
    monkeypatch.setattr(ibis.options.repr.interactive, "max_columns", 3)
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_2 " in text
    assert " c_3 " not in text

    # max_columns = None
    monkeypatch.setattr(ibis.options.repr.interactive, "max_columns", None)
    text = "".join(s.text for s in console.render(expr, options))
    assert " c_0 " in text
    if is_jupyter:
        # All columns written
        assert " c_49 " in text
    else:
        # width calculations still truncates
        assert " c_19 " not in text


@pytest.mark.parametrize("expr_type", ["table", "column"])
@pytest.mark.parametrize("interactive", [True, False])
def test_repr_mimebundle(alltypes, interactive, expr_type, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", interactive)

    if expr_type == "column":
        expr = alltypes.date_string_col
    else:
        expr = alltypes.select("date_string_col")

    reprs = expr._repr_mimebundle_(include=["text/plain", "text/html"], exclude=[])
    for format in ["text/plain", "text/html"]:
        if interactive:
            assert "r0.date_string_col" not in reprs[format]
        else:
            assert "r0.date_string_col" in reprs[format]


@pytest.mark.never(
    ["postgres", "bigquery", "duckdb"],
    reason="These backends explicitly do support Geo operations",
)
@pytest.mark.parametrize("op", [ops.GeoDistance, ops.GeoAsText, ops.GeoUnaryUnion])
def test_has_operation_no_geo(con, op):
    """Previously some backends mistakenly reported Geo operations as
    supported.

    Since most backends don't support Geo operations, we test that
    they're excluded here, skipping the few backends that explicitly do
    support them.
    """
    assert not con.has_operation(op)


@pytest.mark.parametrize(
    ("module_name", "op"),
    [
        param(backend, obj, marks=getattr(mark, backend), id=f"{backend}-{name}")
        for name, obj in sorted(inspect.getmembers(builtins), key=itemgetter(0))
        for backend in sorted(ALL_BACKENDS)
        # filter out builtins that are types, except for tuples on ClickHouse
        # and duckdb because tuples are used to represent lists of expressions
        if isinstance(obj, type)
        if (obj != tuple or backend not in ("clickhouse", "duckdb"))
        if (backend != "pyspark" or vparse(pd.__version__) < vparse("2"))
    ],
)
def test_has_operation_no_builtins(module_name, op):
    mod = importlib.import_module(f"ibis.backends.{module_name}")
    assert not mod.Backend.has_operation(op)


def test_get_backend(con, alltypes, monkeypatch):
    assert ibis.get_backend(alltypes) is con
    assert ibis.get_backend(alltypes.id.min()) is con

    with pytest.raises(com.IbisError, match="contains unbound tables"):
        ibis.get_backend(ibis.table({"x": "int"}))

    monkeypatch.setattr(ibis.options, "default_backend", con)
    assert ibis.get_backend() is con
    expr = ibis.literal(1) + 2
    assert ibis.get_backend(expr) is con


def test_set_backend(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", None)
    ibis.set_backend(con)
    assert ibis.get_backend() is con


@pytest.mark.parametrize(
    "name",
    [
        param(name, marks=getattr(mark, name), id=name)
        for name in ("datafusion", "duckdb", "polars", "sqlite", "pandas", "dask")
    ],
)
def test_set_backend_name(name, monkeypatch):
    # Don't need to test with all backends, only checking that things are
    # plumbed through correctly.
    monkeypatch.setattr(ibis.options, "default_backend", None)
    ibis.set_backend(name)
    assert ibis.get_backend().name == name


@pytest.mark.parametrize(
    "url",
    [
        param(
            "clickhouse://default@localhost:8123",
            marks=mark.clickhouse,
            id="clickhouse",
        ),
        param(
            "mysql://ibis:ibis@localhost:3306",
            marks=mark.mysql,
            id="mysql",
        ),
        param(
            "postgres://postgres:postgres@localhost:5432",
            marks=mark.postgres,
            id="postgres",
        ),
    ],
)
def test_set_backend_url(url, monkeypatch):
    # Don't need to test with all backends, only checking that things are
    # plumbed through correctly.
    monkeypatch.setattr(ibis.options, "default_backend", None)
    name = url.split("://")[0]
    ibis.set_backend(url)
    assert ibis.get_backend().name == name


@pytest.mark.notyet(
    [
        "bigquery",
        "dask",
        "datafusion",
        "duckdb",
        "exasol",
        "impala",
        "mssql",
        "mysql",
        "pandas",
        "polars",
        "postgres",
        "risingwave",
        "pyspark",
        "sqlite",
    ],
    reason="backend doesn't support timestamp with scale parameter",
)
@pytest.mark.notimpl(["clickhouse"], reason="create table isn't implemented")
@pytest.mark.notimpl(
    ["snowflake"], reason="scale not implemented in ibis's snowflake backend"
)
@pytest.mark.broken(
    ["oracle"], reason="oracle doesn't allow DESCRIBE outside of its CLI"
)
@pytest.mark.broken(["druid"], reason="dialect is broken")
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
def test_create_table_timestamp(con, temp_table):
    schema = ibis.schema(
        dict(zip(string.ascii_letters, map("timestamp({:d})".format, range(10))))
    )
    con.create_table(
        temp_table,
        schema=schema,
        overwrite=True,
    )
    rows = con.raw_sql(f"DESCRIBE {temp_table}").fetchall()
    result = ibis.schema((name, typ) for name, typ, *_ in rows)
    assert result == schema


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_ref_count(backend, con, alltypes):
    non_persisted_table = alltypes.mutate(test_column="calculation")
    persisted_table = non_persisted_table.cache()

    op = non_persisted_table.op()

    # ref count is unaffected without a context manager
    assert con._query_cache.refs[op] == 1
    backend.assert_frame_equal(
        non_persisted_table.to_pandas(), persisted_table.to_pandas()
    )
    assert con._query_cache.refs[op] == 1


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression(backend, alltypes):
    non_persisted_table = alltypes.mutate(test_column="calculation", other_calc="xyz")
    persisted_table = non_persisted_table.cache()
    backend.assert_frame_equal(
        non_persisted_table.to_pandas(), persisted_table.to_pandas()
    )


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_contextmanager(backend, alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc"
    )
    with non_cached_table.cache() as cached_table:
        backend.assert_frame_equal(
            non_cached_table.to_pandas(), cached_table.to_pandas()
        )


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_contextmanager_ref_count(backend, con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc 2"
    )
    op = non_cached_table.op()
    with non_cached_table.cache() as cached_table:
        backend.assert_frame_equal(
            non_cached_table.to_pandas(), cached_table.to_pandas()
        )
        assert con._query_cache.refs[op] == 1
    assert con._query_cache.refs[op] == 0


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
def test_persist_expression_multiple_refs(backend, con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc 2"
    )
    op = non_cached_table.op()
    with non_cached_table.cache() as cached_table:
        backend.assert_frame_equal(
            non_cached_table.to_pandas(), cached_table.to_pandas()
        )

        name1 = cached_table.op().name

        with non_cached_table.cache() as nested_cached_table:
            name2 = nested_cached_table.op().name
            assert not nested_cached_table.to_pandas().empty

            # there are two refs to the uncached expression
            assert con._query_cache.refs[op] == 2

        # one ref to the uncached expression was removed by the context manager
        assert con._query_cache.refs[op] == 1

    # no refs left after the outer context manager exits
    assert con._query_cache.refs[op] == 0

    # assert that tables have been dropped
    assert name1 not in con.list_tables()
    assert name2 not in con.list_tables()


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_repeated_cache(alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc 2"
    )
    with non_cached_table.cache() as cached_table:
        with cached_table.cache() as nested_cached_table:
            assert not nested_cached_table.to_pandas().empty


@mark.notimpl(["datafusion", "flink", "impala", "trino", "druid"])
@mark.never(
    ["mssql"],
    reason="mssql supports support temporary tables through naming conventions",
)
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
@mark.notimpl(
    ["oracle"],
    reason="Oracle error message for a missing table/view doesn't include the name of the table",
)
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
def test_persist_expression_release(con, alltypes):
    non_cached_table = alltypes.mutate(
        test_column="calculation", other_column="big calc 3"
    )
    cached_table = non_cached_table.cache()
    cached_table.release()
    assert con._query_cache.refs[non_cached_table.op()] == 0

    with pytest.raises(
        com.IbisError,
        match=r".+Did you call `\.release\(\)` twice on the same expression\?",
    ):
        cached_table.release()

    with pytest.raises(Exception, match=cached_table.op().name):
        cached_table.execute()


@contextlib.contextmanager
def gen_test_name(con: BaseBackend) -> str:
    name = gen_name("test_table")
    try:
        yield name
    finally:
        con.drop_table(name, force=True)


@mark.notimpl(
    ["polars"],
    raises=NotImplementedError,
    reason="overwriting not implemented in ibis for this backend",
)
@mark.broken(
    ["druid"], raises=NotImplementedError, reason="generated SQL fails to parse"
)
@mark.notimpl(["impala"], reason="impala doesn't support memtable")
@mark.notimpl(["pyspark"])
@mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason=(
        "Unsupported operation: <class 'ibis.expr.operations.relations.Selection'>. "
        "If `obj` is of `ir.Table`, the operation must be `InMemoryTable`."
    ),
)
def test_overwrite(ddl_con, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", ddl_con)

    t0 = ibis.memtable({"a": [1, 2, 3]})

    with gen_test_name(ddl_con) as x:
        t1 = ddl_con.create_table(x, t0)
        t2 = t1.filter(t1.a < 3)
        expected_count = 2

        assert t2.count().execute() == expected_count

        with gen_test_name(ddl_con) as y:
            assert ddl_con.create_table(y, t2).count().execute() == expected_count

        assert (
            ddl_con.create_table(x, t2, overwrite=True).count().execute()
            == expected_count
        )
        assert t2.count().execute() == expected_count


@pytest.mark.notyet(["datafusion"], reason="cannot list or drop databases")
def test_create_database(con_create_database):
    database = gen_name("test_create_database")
    con_create_database.create_database(database)
    assert database in con_create_database.list_databases()
    con_create_database.drop_database(database)
    assert database not in con_create_database.list_databases()


def test_create_schema(con_create_schema):
    schema = gen_name("test_create_schema")
    con_create_schema.create_schema(schema)
    assert schema in con_create_schema.list_schemas()
    con_create_schema.drop_schema(schema)
    assert schema not in con_create_schema.list_schemas()


@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: information_schema.schemata is not supported,",
)
def test_list_schemas(con_create_schema):
    schemas = con_create_schema.list_schemas()
    assert len(schemas) == len(set(schemas))


@pytest.mark.notyet(["datafusion"], reason="cannot list or drop databases")
def test_create_database_schema(con_create_database_schema):
    database = gen_name("test_create_database")
    con_create_database_schema.create_database(database)
    try:
        schema = gen_name("test_create_database_schema")
        con_create_database_schema.create_schema(schema, database=database)
        con_create_database_schema.drop_schema(schema, database=database)
    finally:
        con_create_database_schema.drop_database(database)


@pytest.mark.notyet(["datafusion"], reason="cannot list or drop databases")
def test_list_databases_schemas(con_create_database_schema):
    database = gen_name("test_create_database")
    con_create_database_schema.create_database(database)
    try:
        schema = gen_name("test_create_database_schema")
        con_create_database_schema.create_schema(schema, database=database)

        try:
            assert schema in con_create_database_schema.list_schemas(database=database)
        finally:
            con_create_database_schema.drop_schema(schema, database=database)
    finally:
        con_create_database_schema.drop_database(database)


@pytest.mark.notyet(
    ["pandas", "dask", "polars", "datafusion"],
    reason="this is a no-op for in-memory backends",
)
@pytest.mark.notyet(
    ["trino", "clickhouse", "impala", "bigquery", "flink"],
    reason="Backend client does not conform to DB-API, subsequent op does not raise",
)
@pytest.mark.skip()
def test_close_connection(con):
    if con.name == "pyspark":
        # It would be great if there were a simple way to say "give me a new
        # spark context" but I haven't found it.
        pytest.skip("Closing spark context breaks subsequent tests")
    new_con = getattr(ibis, con.name).connect(*con._con_args, **con._con_kwargs)

    # Run any command that hits the backend
    _ = new_con.list_tables()
    new_con.disconnect()

    # DB-API states that subsequent execution attempt should raise
    with pytest.raises(Exception):  # noqa:B017
        new_con.list_tables()


@pytest.mark.notyet(
    ["clickhouse"],
    raises=AttributeError,
    reason="JSON extension is experimental and not enabled by default in testing",
)
@pytest.mark.notyet(
    ["datafusion", "polars", "mssql", "druid", "oracle", "exasol", "impala"],
    raises=AttributeError,
    reason="JSON type not implemented",
)
@pytest.mark.notimpl(
    ["risingwave", "sqlite"],
    raises=pa.ArrowTypeError,
    reason="mismatch between output value and expected input type",
)
@pytest.mark.never(
    ["snowflake"],
    raises=TypeError,
    reason="snowflake uses a custom pyarrow extension type for JSON pretty printing",
)
def test_json_to_pyarrow(con):
    t = con.tables.json_t
    table = t.to_pyarrow()
    js = table["js"]

    expected = [
        {"a": [1, 2, 3, 4], "b": 1},
        {"a": None, "b": 2},
        {"a": "foo", "c": None},
        None,
        [42, 47, 55],
        [],
    ]
    expected = {json.dumps(val) for val in expected}

    result = {
        # loads and dumps so the string representation is the same
        json.dumps(json.loads(val))
        for val in js.to_pylist()
    }
    assert result == expected
