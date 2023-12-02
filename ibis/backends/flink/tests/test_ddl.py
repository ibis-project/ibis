from __future__ import annotations

import os
import tempfile

import pandas as pd
import pyarrow as pa
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.conftest import TEST_TABLES

try:
    from py4j.protocol import Py4JJavaError
except ImportError:
    Py4JJavaError = None

_awards_players_schema = sch.Schema(
    {
        "playerID": dt.string,
        "awardID": dt.string,
        "yearID": dt.int32,
        "lgID": dt.string,
        "tie": dt.string,
        "notes": dt.string,
    }
)

_functiona_alltypes_schema = sch.Schema(
    {
        "id": dt.int32,
        "bool_col": dt.bool,
        "smallint_col": dt.int16,
        "int_col": dt.int32,
        "bigint_col": dt.int64,
        "float_col": dt.float32,
        "double_col": dt.float64,
        "date_string_col": dt.string,
        "string_col": dt.string,
        "timestamp_col": dt.timestamp(scale=3),
        "year": dt.int32,
        "month": dt.int32,
    }
)


@pytest.fixture(autouse=True)
def reset_con(con):
    yield
    tables_to_drop = list(set(con.list_tables()) - set(TEST_TABLES.keys()))
    for table in tables_to_drop:
        con.drop_table(table, force=True)


@pytest.fixture
def awards_players_schema():
    return _awards_players_schema


@pytest.fixture
def functiona_alltypes_schema():
    return _functiona_alltypes_schema


@pytest.fixture
def csv_source_configs():
    def generate_csv_configs(csv_file):
        return {
            "connector": "filesystem",
            "path": f"ci/ibis-testing-data/csv/{csv_file}.csv",
            "format": "csv",
            "csv.ignore-parse-errors": "true",
        }

    return generate_csv_configs


@pytest.fixture
def tempdir_sink_configs():
    def generate_tempdir_configs(tempdir):
        return {"connector": "filesystem", "path": tempdir, "format": "csv"}

    return generate_tempdir_configs


def test_list_tables(con):
    assert len(con.list_tables())
    assert con.list_tables(catalog="default_catalog", database="default_database")


def test_create_table_from_schema(
    con, awards_players_schema, temp_table, csv_source_configs
):
    new_table = con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=csv_source_configs("awards_players"),
    )
    assert temp_table in con.list_tables()
    assert new_table.schema() == awards_players_schema


@pytest.mark.parametrize("temp", [True, False])
def test_create_table(con, awards_players_schema, temp_table, csv_source_configs, temp):
    con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=csv_source_configs("awards_players"),
        temp=temp,
    )
    assert temp_table in con.list_tables()

    if temp:
        with pytest.raises(Py4JJavaError):
            con.drop_table(temp_table)

    con.drop_table(temp_table, temp=temp)

    assert temp_table not in con.list_tables()


def test_recreate_table_from_schema(
    con, awards_players_schema, temp_table, csv_source_configs
):
    # create table once
    new_table = con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=csv_source_configs("awards_players"),
    )
    assert temp_table in con.list_tables()
    assert new_table.schema() == awards_players_schema

    # create the same table a second time should fail
    with pytest.raises(
        Py4JJavaError,
        match="org.apache.flink.table.catalog.exceptions.TableAlreadyExistException",
    ):
        new_table = con.create_table(
            temp_table,
            schema=awards_players_schema,
            tbl_properties=csv_source_configs("awards_players"),
            overwrite=False,
        )


def test_force_recreate_table_from_schema(
    con, awards_players_schema, temp_table, csv_source_configs
):
    # create table once
    new_table = con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=csv_source_configs("awards_players"),
    )
    assert temp_table in con.list_tables()
    assert new_table.schema() == awards_players_schema

    # force creating the same twice a second time
    new_table = con.create_table(
        temp_table,
        schema=awards_players_schema,
        tbl_properties=csv_source_configs("awards_players"),
        overwrite=True,
    )
    assert temp_table in con.list_tables()
    assert new_table.schema() == awards_players_schema


@pytest.mark.parametrize(
    "employee_df",
    [
        pd.DataFrame(
            [("fred flintstone", "award", 2002, "lg_id", "tie", "this is a note")]
        )
    ],
)
@pytest.mark.parametrize(
    "schema_props", [(None, None), (_awards_players_schema, "awards_players")]
)
def test_recreate_in_mem_table(
    con, employee_df, schema_props, temp_table, csv_source_configs
):
    # create table once
    schema = schema_props[0]
    if schema_props[1] is not None:
        tbl_properties = csv_source_configs(schema_props[1])
    else:
        tbl_properties = None

    new_table = con.create_table(
        name=temp_table,
        obj=employee_df,
        schema=schema,
        tbl_properties=tbl_properties,
    )
    assert temp_table in con.list_tables()
    if schema is not None:
        assert new_table.schema() == schema

    # create the same table a second time should fail
    with pytest.raises(
        Py4JJavaError,
        match="An error occurred while calling o8.createTemporaryView",
    ):
        new_table = con.create_table(
            name=temp_table,
            obj=employee_df,
            schema=schema,
            tbl_properties=tbl_properties,
            overwrite=False,
        )


@pytest.mark.parametrize(
    "employee_df",
    [
        pd.DataFrame(
            [("fred flintstone", "award", 2002, "lg_id", "tie", "this is a note")]
        )
    ],
)
@pytest.mark.parametrize(
    "schema_props", [(None, None), (_awards_players_schema, "awards_players")]
)
def test_force_recreate_in_mem_table(
    con, employee_df, schema_props, temp_table, csv_source_configs
):
    # create table once
    schema = schema_props[0]
    if schema_props[1] is not None:
        tbl_properties = csv_source_configs(schema_props[1])
    else:
        tbl_properties = None

    new_table = con.create_table(
        name=temp_table,
        obj=employee_df,
        schema=schema,
        tbl_properties=tbl_properties,
    )
    assert temp_table in con.list_tables()
    assert temp_table in con.list_views()
    if schema is not None:
        assert new_table.schema() == schema

    # force recreate the same table a second time should succeed
    new_table = con.create_table(
        name=temp_table,
        obj=employee_df,
        schema=schema,
        tbl_properties=tbl_properties,
        overwrite=True,
    )
    assert temp_table in con.list_tables()
    assert temp_table in con.list_views()
    if schema is not None:
        assert new_table.schema() == schema


def test_create_source_table_with_watermark(
    con, functiona_alltypes_schema, temp_table, csv_source_configs
):
    new_table = con.create_table(
        temp_table,
        schema=functiona_alltypes_schema,
        tbl_properties=csv_source_configs("functional_alltypes"),
        watermark=ibis.watermark(
            time_col="timestamp_col", allowed_delay=ibis.interval(seconds=15)
        ),
    )
    assert temp_table in con.list_tables()
    assert new_table.schema() == functiona_alltypes_schema


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(
            [("fred flintstone", 35, 1.28), ("barney rubble", 32, 2.32)], id="list"
        ),
        pytest.param(
            {
                "name": ["fred flintstone", "barney rubble"],
                "age": [35, 32],
                "gpa": [1.28, 2.32],
            },
            id="dict",
        ),
        pytest.param(
            pd.DataFrame(
                [("fred flintstone", 35, 1.28), ("barney rubble", 32, 2.32)],
                columns=["name", "age", "gpa"],
            ),
            id="pandas_dataframe",
        ),
        pytest.param(
            pa.Table.from_arrays(
                [
                    pa.array(["fred flintstone", "barney rubble"]),
                    pa.array([35, 32]),
                    pa.array([1.28, 2.32]),
                ],
                names=["name", "age", "gpa"],
            ),
            id="pyarrow_table",
        ),
    ],
)
def test_insert_values_into_table(con, tempdir_sink_configs, obj):
    sink_schema = sch.Schema({"name": dt.string, "age": dt.int64, "gpa": dt.float64})
    with tempfile.TemporaryDirectory() as tempdir:
        con.create_table(
            "tempdir_sink",
            schema=sink_schema,
            tbl_properties=tempdir_sink_configs(tempdir),
        )
        con.insert("tempdir_sink", obj).wait()
        temporary_file = next(iter(os.listdir(tempdir)))
        with open(os.path.join(tempdir, temporary_file)) as f:
            assert f.read() == '"fred flintstone",35,1.28\n"barney rubble",32,2.32\n'


def test_insert_simple_select(con, tempdir_sink_configs):
    con.create_table(
        "source",
        pd.DataFrame(
            [("fred flintstone", 35, 1.28), ("barney rubble", 32, 2.32)],
            columns=["name", "age", "gpa"],
        ),
    )
    sink_schema = sch.Schema({"name": dt.string, "age": dt.int64})
    source_table = ibis.table(
        sch.Schema({"name": dt.string, "age": dt.int64, "gpa": dt.float64}), "source"
    )
    with tempfile.TemporaryDirectory() as tempdir:
        con.create_table(
            "tempdir_sink",
            schema=sink_schema,
            tbl_properties=tempdir_sink_configs(tempdir),
        )
        con.insert("tempdir_sink", source_table[["name", "age"]]).wait()
        temporary_file = next(iter(os.listdir(tempdir)))
        with open(os.path.join(tempdir, temporary_file)) as f:
            assert f.read() == '"fred flintstone",35\n"barney rubble",32\n'
