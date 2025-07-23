from __future__ import annotations

import builtins
import contextlib
import importlib
import inspect
import json
import os
import re
import string
import subprocess
import sys
from operator import itemgetter
from typing import TYPE_CHECKING

import pytest
import rich.console
import sqlglot as sg
import toolz
from packaging.version import parse as vparse
from pytest import mark, param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.conftest import ALL_BACKENDS
from ibis.backends.tests.errors import (
    DatabricksServerOperationError,
    ExaQueryError,
    ImpalaHiveServer2Error,
    OracleDatabaseError,
    PsycoPg2InternalError,
    PsycoPgUndefinedObject,
    Py4JJavaError,
    PyAthenaDatabaseError,
    PyDruidProgrammingError,
    PyODBCProgrammingError,
    SnowflakeProgrammingError,
)
from ibis.util import gen_name

if TYPE_CHECKING:
    from ibis.backends import BaseBackend


np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")


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
    elif con.name == "athena":
        pytest.xfail("create table must specific external location")

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
        dict(first_name="string", salary="float64"),
        dict(first_name="string", salary="float64").items(),
        ibis.schema(dict(first_name="string", salary="float64")),
    ],
    ids=["no_schema", "dict_schema", "tuples", "schema"],
)
@pytest.mark.notimpl(["druid"])
@pytest.mark.notimpl(
    ["flink"],
    reason="Flink backend supports creating only TEMPORARY VIEW for in-memory data.",
)
def test_create_table(backend, con, temp_table, func, sch):
    df = pd.DataFrame({"first_name": ["A", "B", "C"], "salary": [100.0, 200.0, 300.0]})

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
                    [
                        "pyspark",
                        "trino",
                        "exasol",
                        "risingwave",
                        "impala",
                        "databricks",
                        "athena",
                    ],
                    reason="No support for temp tables",
                ),
                pytest.mark.notyet(
                    ["mssql"],
                    reason="Can't rename temp tables",
                    raises=ValueError,
                ),
                pytest.mark.notimpl(
                    ["bigquery"],
                    reason="tables created with temp=True cause a 404 on retrieval",
                ),
            ],
        ),
        param(
            False,
            True,
            id="no temp, overwrite",
            marks=[
                pytest.mark.notyet(["flink", "polars"]),
                pytest.mark.notyet(
                    ["athena"],
                    raises=(PyAthenaDatabaseError, com.UnsupportedOperationError),
                    reason="quotes are incorrect",
                ),
            ],
        ),
        param(
            True,
            False,
            id="temp, no overwrite",
            marks=[
                pytest.mark.notyet(
                    [
                        "pyspark",
                        "trino",
                        "exasol",
                        "risingwave",
                        "impala",
                        "databricks",
                        "athena",
                    ],
                    reason="No support for temp tables",
                ),
                pytest.mark.notimpl(["mssql"], reason="Incorrect temp table syntax"),
                pytest.mark.notimpl(
                    ["bigquery"],
                    reason="tables created with temp=True cause a 404 on retrieval",
                ),
            ],
        ),
    ],
)
@pytest.mark.notimpl(["druid"])
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
@pytest.mark.notyet(
    ["athena"],
    raises=com.UnsupportedOperationError,
    reason="no reasonable implementation is supported by the database",
)
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
            lambda t: t.select(t.string_col, t.bigint_col),
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
    assert schema.names == expected.names
    # can't directly compare because string lengths may differ across backends
    assert all(type(schema[name]) is type(expected[name]) for name in schema.names)


_LIMIT = {
    "oracle": "FETCH FIRST 10 ROWS ONLY",
}


@pytest.mark.notimpl(["mssql"])
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


@mark.notimpl(
    ["oracle"],
    raises=AssertionError,
    reason="oracle temp tables aren't cleaned up on reconnect because we use global temporary tables",
)
@mark.notimpl(["trino", "druid", "athena"], reason="doesn't implement temporary tables")
@mark.notimpl(["exasol"], reason="Exasol does not support temporary tables")
@pytest.mark.notimpl(
    ["impala", "pyspark"],
    reason="temporary tables not implemented",
    raises=NotImplementedError,
)
@pytest.mark.never(
    ["risingwave", "databricks"],
    raises=com.UnsupportedOperationError,
    reason="Feature is not yet implemented: CREATE TEMPORARY TABLE",
)
@pytest.mark.notyet(
    ["datafusion"],
    raises=Exception,
    reason="temp tables are not supported upstream in datafusion",
)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
def test_create_temporary_table_from_schema(con_no_data, new_schema):
    if con_no_data.name == "snowflake" and os.environ.get("SNOWFLAKE_SNOWPARK"):
        with pytest.raises(
            com.IbisError, match="Cannot reconnect to unconfigured .+ backend"
        ):
            con_no_data.reconnect()
        return

    temp_table = gen_name(f"test_{con_no_data.name}_tmp")
    table = con_no_data.create_table(temp_table, schema=new_schema, temp=True)

    # verify table exist in the current session
    backend_mapping = backend_type_mapping.get(con_no_data.name, dict())
    for column_name, column_type in table.schema().items():
        assert (
            backend_mapping.get(new_schema[column_name], new_schema[column_name])
            == column_type
        )

    if con_no_data.name != "pyspark":
        con_no_data.disconnect()
    con_no_data.reconnect()
    # verify table no longer exist after reconnect
    assert temp_table not in con_no_data.list_tables()


@mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "druid",
        "duckdb",
        "exasol",
        "mssql",
        "mysql",
        "oracle",
        "polars",
        "postgres",
        "risingwave",
        "snowflake",
        "sqlite",
        "trino",
        "athena",
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


@mark.notimpl(["polars", "druid", "athena"])
@mark.never(["impala", "pyspark"], reason="No non-nullable datatypes")
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


@mark.notimpl(["druid"])
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
def test_employee_schema() -> ibis.schema:
    return ibis.schema(
        {
            "first_name": "string",
            "last_name": "string",
            "department_name": "string",
            "salary": "float64",
        }
    )


@pytest.fixture
def employee_empty_temp_table(backend, con, test_employee_schema):
    temp_table_name = gen_name("temp_employee_empty_table")
    _create_temp_table_with_schema(backend, con, temp_table_name, test_employee_schema)
    yield temp_table_name
    con.drop_table(temp_table_name, force=True)


@pytest.fixture
def employee_data_1_temp_table(backend, con, test_employee_schema):
    test_employee_data_1 = pd.DataFrame(
        {
            "first_name": ["A", "B", "C"],
            "last_name": ["D", "E", "F"],
            "department_name": ["AA", "BB", "CC"],
            "salary": [100.0, 200.0, 300.0],
        }
    )

    temp_table_name = gen_name("temp_employee_data_1")
    _create_temp_table_with_schema(
        backend, con, temp_table_name, test_employee_schema, data=test_employee_data_1
    )
    assert temp_table_name in con.list_tables()
    yield temp_table_name
    con.drop_table(temp_table_name, force=True)


@pytest.fixture
def test_employee_data_2():
    import pandas as pd

    df2 = pd.DataFrame(
        {
            "first_name": ["X", "Y", "Z"],
            "last_name": ["A", "B", "C"],
            "department_name": ["XX", "YY", "ZZ"],
            "salary": [400.0, 500.0, 600.0],
        }
    )

    return df2


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


@pytest.mark.notimpl(["polars"], reason="`insert` method not implemented")
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


@pytest.mark.notimpl(["polars"], reason="`insert` method not implemented")
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="truncate not supported upstream",
)
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notyet(
    ["athena"], raises=com.UnsupportedOperationError, reason="s3 location required"
)
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


@pytest.mark.notimpl(["polars"], reason="`insert` method not implemented")
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


@pytest.mark.notimpl(["polars"], reason="`insert` method not implemented")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="DELETE DML not implemented upstream"
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


@pytest.mark.notimpl(["polars"], reason="`insert` method not implemented")
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
    ["polars"], raises=AttributeError, reason="`insert` method not implemented"
)
@pytest.mark.notyet(["druid"], raises=NotImplementedError)
@pytest.mark.notimpl(["polars"], raises=AttributeError)
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
    [
        "bigquery",
        "clickhouse",
        "druid",
        "exasol",
        "impala",
        "mysql",
        "oracle",
        "polars",
        "flink",
        "sqlite",
    ],
    raises=AttributeError,
    reason="doesn't support the common notion of a catalog",
)
@pytest.mark.xfail_version(pyspark=["pyspark<3.4"])
def test_list_catalogs(con):
    # Every backend has its own databases
    test_catalogs = {
        "datafusion": {"datafusion"},
        "duckdb": {"memory"},
        "exasol": set(),
        "flink": set(),
        "mssql": {"ibis-testing"},
        "oracle": set(),
        "postgres": {"postgres", "ibis_testing"},
        "risingwave": {"dev"},
        "snowflake": {"IBIS_TESTING"},
        "trino": {"memory"},
        "pyspark": {"spark_catalog"},
        "databricks": {"hive_metastore", "ibis", "ibis_testing", "samples", "system"},
        "athena": {"AwsDataCatalog"},
    }
    result = set(con.list_catalogs())
    assert test_catalogs[con.name] <= result


@pytest.mark.notyet(
    ["druid", "polars"],
    raises=AttributeError,
    reason="doesn't support the common notion of a database",
)
def test_list_database_contents(con):
    # Every backend has its own databases
    test_databases = {
        "bigquery": {"ibis_gbq_testing"},
        "clickhouse": {"system", "default", "ibis_testing"},
        "datafusion": {"public"},
        "duckdb": {"pg_catalog", "main", "information_schema"},
        "exasol": {"EXASOL"},
        "flink": {"default_database"},
        "impala": {"ibis_testing", "default", "_impala_builtins"},
        "mssql": {"INFORMATION_SCHEMA", "dbo", "guest"},
        "mysql": {"ibis-testing", "information_schema"},
        "oracle": {"SYS", "IBIS"},
        "postgres": {"public", "information_schema"},
        "pyspark": set(),
        "risingwave": {"public", "rw_catalog", "information_schema"},
        "snowflake": {"IBIS_TESTING"},
        "sqlite": {"main"},
        "trino": {"default", "information_schema"},
        "databricks": {"default"},
        "athena": set(),
    }
    result = set(con.list_databases())
    assert test_databases[con.name] <= result


@pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError)
@pytest.mark.notyet(["pyspark"], raises=com.IbisTypeError)
@pytest.mark.notyet(["databricks"], raises=DatabricksServerOperationError)
@pytest.mark.notyet(["bigquery"], raises=com.UnsupportedBackendType)
@pytest.mark.notyet(
    ["postgres"], raises=PsycoPgUndefinedObject, reason="no unsigned int types"
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
    ["athena"], raises=com.UnsupportedOperationError, reason="no temp tables"
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
            "clickhouse://ibis@localhost:8123/default",
            marks=mark.clickhouse,
            id="clickhouse",
        ),
        param("datafusion://", marks=mark.datafusion, id="datafusion"),
        param("impala://localhost:21050/default", marks=mark.impala, id="impala"),
        param("mysql://ibis:ibis@localhost:3306", marks=mark.mysql, id="mysql"),
        param("polars://", marks=mark.polars, id="polars"),
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
    try:
        assert con.execute(ibis.literal(1)) == 1
    finally:
        if con.name != "pyspark":
            con.disconnect()


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
    "obj, table_name",
    [
        param(lambda: pa.table({"a": ["a"], "b": [1]}), "df_arrow", id="pyarrow table"),
        param(
            lambda: pa.table({"a": ["a"], "b": [1]}).to_reader(),
            "df_arrow_batch_reader",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "duckdb",
                        "exasol",
                        "impala",
                        "mssql",
                        "mysql",
                        "oracle",
                        "postgres",
                        "pyspark",
                        "risingwave",
                        "snowflake",
                        "sqlite",
                        "trino",
                        "databricks",
                        "athena",
                    ]
                )
            ],
            id="pyarrow_rbr",
        ),
        param(
            lambda: pa.table({"a": ["a"], "b": [1]}).to_batches()[0],
            "df_arrow_single_batch",
            id="pyarrow_single_batch",
        ),
        param(
            lambda: ds.dataset(pa.table({"a": ["a"], "b": [1]})),
            "df_arrow_dataset",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "exasol",
                        "impala",
                        "mssql",
                        "mysql",
                        "oracle",
                        "postgres",
                        "pyspark",
                        "risingwave",
                        "snowflake",
                        "sqlite",
                        "trino",
                        "databricks",
                        "athena",
                    ],
                    raises=com.UnsupportedOperationError,
                    reason="we don't materialize datasets to avoid perf footguns",
                ),
                pytest.mark.notimpl(["polars"], raises=NotImplementedError),
            ],
            id="pyarrow dataset",
        ),
        param(lambda: pd.DataFrame({"a": ["a"], "b": [1]}), "df_pandas", id="pandas"),
        param(
            lambda: pytest.importorskip("polars").DataFrame({"a": ["a"], "b": [1]}),
            "df_polars_eager",
            id="polars dataframe",
        ),
        param(
            lambda: pytest.importorskip("polars").LazyFrame({"a": ["a"], "b": [1]}),
            "df_polars_lazy",
            id="polars lazyframe",
        ),
        param(
            lambda: ibis.memtable([("a", 1)], columns=["a", "b"]),
            "memtable",
            id="memtable_list",
        ),
        param(
            lambda: ibis.memtable(pa.table({"a": ["a"], "b": [1]})),
            "memtable_pa",
            id="memtable pyarrow",
        ),
        param(
            lambda: ibis.memtable(pd.DataFrame({"a": ["a"], "b": [1]})),
            "memtable_pandas",
            id="memtable pandas",
        ),
        param(
            lambda: ibis.memtable(
                pytest.importorskip("polars").DataFrame({"a": ["a"], "b": [1]})
            ),
            "memtable_polars_eager",
            id="memtable polars dataframe",
        ),
        param(
            lambda: ibis.memtable(
                pytest.importorskip("polars").LazyFrame({"a": ["a"], "b": [1]})
            ),
            "memtable_polars_lazy",
            id="memtable polars lazyframe",
        ),
    ],
)
@pytest.mark.notimpl(["druid"])
@pytest.mark.notimpl(
    ["flink"],
    reason="Flink backend supports creating only TEMPORARY VIEW for in-memory data.",
)
def test_create_table_in_memory(con, obj, table_name, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    table_name = gen_name(table_name)
    t = con.create_table(table_name, obj())

    try:
        assert table_name in con.list_tables()
        assert pa.table({"a": ["a"], "b": [1]}).equals(t.to_pyarrow())
    finally:
        con.drop_table(table_name, force=True)


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


@pytest.mark.parametrize("show_types", [True, False])
def test_interactive_repr_show_types(alltypes, show_types, monkeypatch):
    pytest.importorskip("rich")

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
    pytest.importorskip("rich")

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
    pytest.importorskip("rich")

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
        if (obj is not tuple or backend not in ("clickhouse", "duckdb"))
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
        for name in ("datafusion", "duckdb", "polars", "sqlite", "pyspark")
    ],
)
def test_set_backend_name(name, monkeypatch):
    # Don't need to test with all backends, only checking that things are
    # plumbed through correctly.
    monkeypatch.setattr(ibis.options, "default_backend", None)
    ibis.set_backend(name)
    try:
        con = ibis.get_backend()
        assert con.name == name
    finally:
        if con.name != "pyspark":
            con.disconnect()


@pytest.mark.parametrize(
    "url",
    [
        param(
            "clickhouse://ibis@localhost:8123",
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
    con = ibis.get_backend()
    try:
        assert con.name == name
    finally:
        con.disconnect()


@pytest.mark.notyet(
    [
        "bigquery",
        "datafusion",
        "duckdb",
        "exasol",
        "impala",
        "mssql",
        "mysql",
        "polars",
        "postgres",
        "risingwave",
        "pyspark",
        "sqlite",
        "databricks",
    ],
    reason="backend doesn't support timestamp with scale parameter",
)
@pytest.mark.notimpl(["clickhouse"], reason="create table isn't implemented")
@pytest.mark.notimpl(
    ["snowflake"], reason="scale not implemented in ibis's snowflake backend"
)
@pytest.mark.never(
    ["oracle"], reason="oracle doesn't allow DESCRIBE outside of its CLI"
)
@pytest.mark.notimpl(["athena"], reason="no overwrite")
@pytest.mark.notimpl(["druid"], reason="dialect is broken")
@pytest.mark.notimpl(
    ["flink"],
    raises=com.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
def test_create_table_timestamp(con, temp_table):
    schema = ibis.schema(
        dict(zip(string.ascii_letters, map("timestamp({:d})".format, range(10))))
    )
    con.create_table(temp_table, schema=schema, overwrite=True)

    with con._safe_raw_sql(f"DESCRIBE {temp_table}") as cur:
        rows = cur.fetchall()

    result = ibis.schema((name, typ) for name, typ, *_ in rows)
    assert result == schema


@contextlib.contextmanager
def gen_test_name(con: BaseBackend):
    name = gen_name("test_table")
    yield name
    con.drop_table(name, force=True)


@mark.notimpl(
    ["druid"], raises=NotImplementedError, reason="generated SQL fails to parse"
)
@mark.notimpl(["athena"], reason="syntax isn't correct; probably a sqlglot issue")
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


@contextlib.contextmanager
def create_and_destroy_db(con):
    con.create_database(dbname := gen_name("db"))
    try:
        yield dbname
    finally:
        con.drop_database(dbname)


# TODO: move this to something like `test_ddl.py`
@pytest.mark.notyet(
    ["flink"],
    reason="unclear whether Flink supports cross catalog/database inserts",
    raises=Py4JJavaError,
)
def test_insert_with_database_specified(con_create_database):
    con = con_create_database

    t = ibis.memtable({"a": [1, 2, 3]})

    with create_and_destroy_db(con) as dbname:
        con.create_table(
            table_name := gen_name("table"),
            obj=t,
            database=dbname,
            temp=con.name == "flink",
        )
        try:
            con.insert(table_name, obj=t, database=dbname)
            assert con.table(table_name, database=dbname).count().to_pandas() == 6
        finally:
            con.drop_table(table_name, database=dbname)


@pytest.mark.notyet(["datafusion"], reason="cannot list or drop catalogs")
def test_create_catalog(con_create_catalog):
    catalog = gen_name("test_create_catalog")
    con_create_catalog.create_catalog(catalog)
    assert catalog in con_create_catalog.list_catalogs()
    con_create_catalog.drop_catalog(catalog)
    assert catalog not in con_create_catalog.list_catalogs()


@pytest.mark.parametrize("catalog", [None, "current_catalog"])
def test_create_database(con_create_database, catalog):
    database = gen_name("test_create_database")
    con_create_database.create_database(database)
    assert database in con_create_database.list_databases()
    if catalog is None:
        catalog = None
    else:
        catalog = getattr(con_create_database, "current_catalog", None)
    con_create_database.drop_database(database, catalog=catalog)
    assert database not in con_create_database.list_databases()


def test_list_databases(con_create_database):
    databases = con_create_database.list_databases()
    assert len(databases) == len(set(databases))


@pytest.mark.notyet(["datafusion"], reason="cannot list or drop databases")
def test_create_catalog_database(con_create_catalog_database):
    catalog = gen_name("test_create_catalog")
    con_create_catalog_database.create_catalog(catalog)
    try:
        database = gen_name("test_create_catalog_database")
        con_create_catalog_database.create_database(database, catalog=catalog)
        con_create_catalog_database.drop_database(database, catalog=catalog)
    finally:
        con_create_catalog_database.drop_catalog(catalog)


@pytest.mark.notyet(["datafusion"], reason="cannot list or drop databases")
def test_list_catalogs_databases(con_create_catalog_database):
    catalog = gen_name("test_create_catalog")
    con_create_catalog_database.create_catalog(catalog)
    try:
        database = gen_name("test_create_catalog_database")
        con_create_catalog_database.create_database(database, catalog=catalog)

        try:
            assert database in con_create_catalog_database.list_databases(
                catalog=catalog
            )
        finally:
            con_create_catalog_database.drop_database(database, catalog=catalog)
    finally:
        con_create_catalog_database.drop_catalog(catalog)


@pytest.mark.notyet(
    ["polars", "datafusion"], reason="this is a no-op for in-memory backends"
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
    ["sqlite"],
    raises=pa.ArrowTypeError,
    reason="mismatch between output value and expected input type",
)
@pytest.mark.never(
    ["snowflake"],
    raises=TypeError,
    reason="snowflake uses a custom pyarrow extension type for JSON pretty printing",
)
@pytest.mark.notimpl(["athena"], raises=AttributeError, reason="not yet implemented")
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
        "a",
        "",
        "b",
        None,
        True,
        False,
        42,
        37.37,
    ]
    expected = {json.dumps(val) for val in expected}

    result = {
        # loads and dumps so the string representation is the same
        json.dumps(json.loads(val))
        for val in js.to_pylist()
        # proper null values must be ignored because they cannot be
        # deserialized as JSON
        #
        # they exist in the json_t table, so the `js` value contains them
        if val is not None
    }
    assert result == expected


@pytest.mark.notyet(
    ["risingwave", "exasol", "databricks"],
    raises=com.UnsupportedOperationError,
    reason="no temp table support",
)
@pytest.mark.notyet(
    ["impala", "trino", "athena"],
    raises=NotImplementedError,
    reason="no temp table support",
)
@pytest.mark.notyet(
    ["druid"], raises=NotImplementedError, reason="doesn't support create_table"
)
@pytest.mark.notyet(
    ["flink"], raises=com.IbisError, reason="no persistent temp table support"
)
def test_schema_with_caching(alltypes):
    t1 = alltypes.limit(5).select("bigint_col", "string_col")
    t2 = alltypes.limit(5).select("string_col", "bigint_col")

    pt1 = t1.cache()
    pt2 = t2.cache()

    assert pt1.schema() == t1.schema()
    assert pt2.schema() == t2.schema()


@pytest.mark.notyet(
    ["druid"], raises=NotImplementedError, reason="doesn't support create_table"
)
@pytest.mark.notyet(["polars"], reason="Doesn't support insert")
@pytest.mark.notyet(
    ["datafusion"], reason="Doesn't support table creation from records"
)
@pytest.mark.notimpl(
    ["flink"], reason="Temp tables are implemented as views, which don't support insert"
)
@pytest.mark.parametrize(
    "first_row, second_row",
    [
        param([{"a": 1, "b": 2}], [{"b": 22, "a": 11}], id="column order reversed"),
        param([{"a": 1, "b": 2}], [{"a": 11, "b": 22}], id="column order matching"),
        param(
            [{"a": 1, "b": 2}],
            [(11, 22)],
            marks=[
                pytest.mark.notimpl(
                    ["impala"],
                    reason="Impala DDL has strict validation checks on schema",
                )
            ],
            id="auto generated cols",
        ),
    ],
)
def test_insert_using_col_name_not_position(con, first_row, second_row, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    table_name = gen_name("table")
    con.create_table(table_name, first_row)
    con.insert(table_name, second_row)

    result = con.table(table_name).order_by("a").to_pyarrow()
    expected_result = pa.table({"a": [1, 11], "b": [2, 22]})

    assert result.equals(expected_result)

    # Ideally we'd use a temp table for this test, but several backends don't
    # support them and it's nice to know that data are being inserted correctly.
    con.drop_table(table_name)


CON_ATTR = {"bigquery": "client", "flink": "_table_env", "pyspark": "_session"}
DEFAULT_CON_ATTR = "con"


@pytest.mark.parametrize("top_level", [True, False])
@pytest.mark.never(["polars"], reason="don't have a connection concept")
def test_from_connection(con, top_level):
    backend = getattr(ibis, con.name) if top_level else type(con)
    new_con = backend.from_connection(getattr(con, CON_ATTR.get(con.name, "con")))
    result = int(new_con.execute(ibis.literal(1, type="int")))
    assert result == 1


def test_table_not_found(con):
    with pytest.raises(com.TableNotFound):
        con.table(gen_name("table_not_found"))


@pytest.mark.notimpl(
    ["flink"], raises=com.IbisError, reason="not yet implemented for Flink"
)
def test_no_accidental_cross_database_table_load(con_create_database):
    con = con_create_database

    # Create an extra database
    con.create_database(dbname := gen_name("dummy_db"))

    # Create table with same name in current db and dummy db
    con.create_table(
        table := gen_name("table"), schema=(sch1 := ibis.schema({"a": "int"}))
    )

    con.create_table(table, schema=ibis.schema({"b": "string"}), database=dbname)

    # Can grab table object from current db:
    t = con.table(table)
    assert t.schema().equals(sch1)

    con.drop_table(table)

    # Now attempting to load same table name without specifying db should fail
    with pytest.raises(com.TableNotFound):
        t = con.table(table)

    # But can load if specify other db
    t = con.table(table, database=dbname)

    # Clean up
    con.drop_table(table, database=dbname)
    con.drop_database(dbname)


@pytest.mark.notyet(["druid"], reason="can't create tables")
@pytest.mark.notyet(
    ["flink"], reason="can't create non-temporary tables from in-memory data"
)
def test_cross_database_join(con_create_database, monkeypatch):
    con = con_create_database

    monkeypatch.setattr(ibis.options, "default_backend", con)

    left = ibis.memtable({"a": [1], "b": [2]})
    right = ibis.memtable({"a": [1], "c": [3]})

    # Create an extra database
    con.create_database(dbname := gen_name("dummy_db"))

    # Insert left into current_database
    left = con.create_table(left_table := gen_name("left"), obj=left)

    # Insert right into new database
    right = con.create_table(
        right_table := gen_name("right"), obj=right, database=dbname
    )

    expr = left.join(right, "a")
    assert expr.columns == ("a", "b", "c")

    result = expr.to_pyarrow()
    expected = pa.Table.from_pydict({"a": [1], "b": [2], "c": [3]})

    assert result.equals(expected)

    con.drop_table(left_table)
    con.drop_table(right_table, database=dbname)
    con.drop_database(dbname)


@pytest.mark.notimpl(
    ["druid"], raises=PyDruidProgrammingError, reason="doesn't implement CREATE syntax"
)
@pytest.mark.notimpl(["clickhouse"], reason="create table isn't implemented")
@pytest.mark.notyet(["flink"], raises=AttributeError, reason="no _safe_raw_sql method")
@pytest.mark.notyet(["polars"], reason="Doesn't support insert")
@pytest.mark.notimpl(
    ["impala", "trino"], reason="Default constraints are not supported"
)
@pytest.mark.notimpl(
    ["databricks"],
    reason="Default constraints ARE supported, "
    "but you have to enable them with a property AND set DEFAULT, so no",
    raises=DatabricksServerOperationError,
)
@pytest.mark.notimpl(["athena"], reason="insert isn't implemented yet")
@pytest.mark.xfail_version(pyspark=["pyspark<3.4"])
def test_insert_into_table_missing_columns(con, temp_table):
    db = getattr(con, "current_database", None)

    raw_ident = sg.table(
        temp_table,
        db=db if db is None else sg.to_identifier(db, quoted=True),
        quoted=True,
    ).sql("duckdb")

    ct_sql = f'CREATE TABLE {raw_ident} ("a" INT DEFAULT 1, "b" INT)'
    sg_expr = sg.parse_one(ct_sql, read="duckdb")
    with con._safe_raw_sql(sg_expr.sql(dialect=con.dialect)):
        pass
    con.insert(temp_table, [{"b": 1}])

    result = con.table(temp_table).to_pyarrow().to_pydict()
    expected_result = {"a": [1], "b": [1]}

    assert result == expected_result


@pytest.mark.notyet(["druid"], raises=AssertionError, reason="can't drop tables")
@pytest.mark.notyet(
    ["clickhouse"], raises=AssertionError, reason="memtables are assembled every time"
)
@pytest.mark.notyet(
    ["bigquery"], raises=AssertionError, reason="test is flaky", strict=False
)
def test_memtable_cleanup(con):
    t = ibis.memtable({"a": [1, 2, 3], "b": list("def")})

    name = t.op().name

    # the table isn't registered until we actually execute, and since we
    # haven't yet executed anything, the table shouldn't be there
    assert name not in con.list_tables()

    # execute, which means the table is registered and should be visible in
    # con.list_tables()
    con.execute(t.select("a"))
    assert name in con.list_tables()

    con.execute(t.select("b"))
    assert name in con.list_tables()


@pytest.mark.notimpl(
    ["clickhouse"],
    raises=AssertionError,
    reason="backend doesn't use _register_in_memory_table",
)
def test_memtable_registered_exactly_once(con, mocker):
    spy = mocker.spy(con, "_register_in_memory_table")

    data = {"a": [1, 2, 3], "b": ["a", "b", "c"]}

    t = ibis.memtable(data)

    assert len(con.execute(t)) == 3
    assert len(con.execute(t)) == 3

    spy.assert_called_once_with(t.op())


@pytest.mark.parametrize("i", range(5))
def test_stateful_data_is_loaded_once(
    con,
    data_dir,
    tmp_path_factory,
    worker_id,
    mocker,
    i,  # noqa: ARG001
):
    TestConf = pytest.importorskip(f"ibis.backends.{con.name}.tests.conftest").TestConf
    if not TestConf.stateful:
        pytest.skip("TestConf is not stateful, skipping test")

    spy = mocker.spy(TestConf, "stateless_load")

    for _ in range(2):
        TestConf.load_data(data_dir, tmp_path_factory, worker_id)

    # also verify that it's been called once, by checking that there's at least
    # one table
    assert con.list_tables()

    # Ensure that the stateful load is called only once the one time it is
    # called is from the `con` input, which *should* work across processes
    spy.assert_not_called()
