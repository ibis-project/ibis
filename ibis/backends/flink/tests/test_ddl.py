from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.errors import Py4JJavaError


@pytest.fixture
def tempdir_sink_configs():
    def generate_tempdir_configs(tempdir):
        return {"connector": "filesystem", "path": tempdir, "format": "csv"}

    return generate_tempdir_configs


@pytest.mark.parametrize("temp", [True, False])
def test_list_tables(con, temp):
    assert len(con.list_tables(temp=temp))
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
    "schema, table_name",
    [(None, None), (TEST_TABLES["awards_players"], "awards_players")],
)
def test_recreate_in_mem_table(con, schema, table_name, temp_table, csv_source_configs):
    employee_df = pd.DataFrame(
        [("fred flintstone", "award", 2002, "lg_id", "tie", "this is a note")]
    )
    # create table once
    if table_name is not None:
        tbl_properties = csv_source_configs(table_name)
    else:
        tbl_properties = None

    new_table = con.create_table(
        name=temp_table,
        obj=employee_df,
        schema=schema,
        tbl_properties=tbl_properties,
        temp=True,
    )
    try:
        assert temp_table in con.list_tables()
        if schema is not None:
            assert new_table.schema() == schema

        # create the same table a second time should fail
        with pytest.raises(
            Py4JJavaError,
            match=r"An error occurred while calling o\d+\.createTemporaryView",
        ):
            new_table = con.create_table(
                name=temp_table,
                obj=employee_df,
                schema=schema,
                tbl_properties=tbl_properties,
                overwrite=False,
                temp=True,
            )
    finally:
        con.drop_table(temp_table, force=True)


@pytest.mark.parametrize(
    "schema_props", [(None, None), (TEST_TABLES["awards_players"], "awards_players")]
)
def test_force_recreate_in_mem_table(con, schema_props, temp_table, csv_source_configs):
    employee_df = pd.DataFrame(
        [("fred flintstone", "award", 2002, "lg_id", "tie", "this is a note")]
    )
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
        temp=True,
    )
    try:
        assert temp_table in con.list_tables()
        if schema is not None:
            assert new_table.schema() == schema

        # force recreate the same table a second time should succeed
        new_table = con.create_table(
            name=temp_table,
            obj=employee_df,
            schema=schema,
            tbl_properties=tbl_properties,
            temp=True,
            overwrite=True,
        )
        assert temp_table in con.list_tables()
        if schema is not None:
            assert new_table.schema() == schema
    finally:
        con.drop_table(temp_table, force=True)


@pytest.fixture
def functional_alltypes_schema_w_nonnullable_columns():
    return sch.Schema(
        {
            "id": dt.int32(nullable=False),
            "bool_col": dt.bool(nullable=False),
            "smallint_col": dt.int16(nullable=False),
            "int_col": dt.int32(nullable=False),
            "bigint_col": dt.int64(nullable=False),
            "float_col": dt.float32(nullable=False),
            "double_col": dt.float64(nullable=False),
            "date_string_col": dt.string(nullable=False),
            "string_col": dt.string(nullable=False),
            "year": dt.int32(nullable=False),
            "month": dt.int32(nullable=False),
            "timestamp_col": dt.timestamp(scale=3),
        }
    )


@pytest.mark.parametrize(
    "primary_key",
    [
        None,
        "id",
        ["id"],
        ["month"],
        ["id", "string_col"],
        ["id", "string_col", "year"],
    ],
)
def test_create_source_table_with_watermark_and_primary_key(
    con,
    temp_table,
    functional_alltypes_schema_w_nonnullable_columns,
    csv_source_configs,
    primary_key,
):
    new_table = con.create_table(
        temp_table,
        schema=functional_alltypes_schema_w_nonnullable_columns,
        tbl_properties=csv_source_configs("functional_alltypes"),
        watermark=ibis.watermark(
            time_col="timestamp_col", allowed_delay=ibis.interval(seconds=15)
        ),
        primary_key=primary_key,
    )
    try:
        assert temp_table in con.list_tables()
        assert new_table.schema() == functional_alltypes_schema_w_nonnullable_columns
    finally:
        con.drop_table(temp_table, force=True)


@pytest.mark.parametrize(
    "primary_key",
    [
        "nonexistent_column",
        ["nonexistent_column"],
        ["id", "nonexistent_column"],
    ],
)
def test_create_table_failure_with_invalid_primary_keys(
    con,
    temp_table,
    functional_alltypes_schema_w_nonnullable_columns,
    csv_source_configs,
    primary_key,
):
    with pytest.raises(exc.IbisError):
        con.create_table(
            temp_table,
            schema=functional_alltypes_schema_w_nonnullable_columns,
            tbl_properties=csv_source_configs("functional_alltypes"),
            primary_key=primary_key,
        )
    assert temp_table not in con.list_tables()


@pytest.fixture
def temp_view(con):
    name = ibis.util.gen_name("view")
    yield name
    con.drop_view(name, force=True)


@pytest.mark.parametrize("temp", [True, False])
def test_create_view(
    con, temp_table, awards_players_schema, csv_source_configs, temp_view, temp
):
    table = con.create_table(
        name=temp_table,
        schema=awards_players_schema,
        tbl_properties=csv_source_configs("awards_players"),
    )
    assert temp_table in con.list_tables()

    con.create_view(
        name=temp_view,
        obj=table,
        force=False,
        temp=temp,
        overwrite=False,
    )
    view_list = sorted(con.list_tables())
    assert temp_view in view_list

    # Try to re-create the same view with `force=False`
    with pytest.raises(Py4JJavaError):
        con.create_view(
            name=temp_view,
            obj=table,
            force=False,
            temp=temp,
            overwrite=False,
        )
    assert view_list == sorted(con.list_tables())

    # Try to re-create the same view with `force=True`
    con.create_view(
        name=temp_view,
        obj=table,
        force=True,
        temp=temp,
        overwrite=False,
    )
    assert view_list == sorted(con.list_tables())

    # Overwrite the view
    con.create_view(
        name=temp_view,
        obj=table,
        force=False,
        temp=temp,
        overwrite=True,
    )
    assert view_list == sorted(con.list_tables())

    con.drop_view(name=temp_view, temp=temp, force=True)
    assert temp_view not in con.list_tables()


def test_rename_table(con, awards_players_schema, temp_table, csv_source_configs):
    table_name = temp_table
    con.create_table(
        name=table_name,
        schema=awards_players_schema,
        tbl_properties=csv_source_configs("awards_players"),
    )
    assert table_name in con.list_tables()

    new_table_name = f"{table_name}_new"
    con.rename_table(
        old_name=table_name,
        new_name=new_table_name,
        force=False,
    )
    assert table_name not in con.list_tables()
    assert new_table_name in con.list_tables()

    con.drop_table(new_table_name)
    assert new_table_name not in con.list_tables()


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
        try:
            con.insert("tempdir_sink", obj).wait()
            temporary_file = next(iter(os.listdir(tempdir)))
            with open(os.path.join(tempdir, temporary_file)) as f:
                assert (
                    f.read() == '"fred flintstone",35,1.28\n"barney rubble",32,2.32\n'
                )
        finally:
            con.drop_table("tempdir_sink", force=True)


def test_insert_simple_select(con, tempdir_sink_configs):
    con.create_table(
        "source",
        pd.DataFrame(
            [("fred flintstone", 35, 1.28), ("barney rubble", 32, 2.32)],
            columns=["name", "age", "gpa"],
        ),
        temp=True,
    )
    try:
        sink_schema = sch.Schema({"name": dt.string, "age": dt.int64})
        source_table = ibis.table(
            sch.Schema({"name": dt.string, "age": dt.int64, "gpa": dt.float64}),
            "source",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            con.create_table(
                "tempdir_sink",
                schema=sink_schema,
                tbl_properties=tempdir_sink_configs(tempdir),
            )
            try:
                con.insert("tempdir_sink", source_table[["name", "age"]]).wait()
                temporary_file = next(iter(os.listdir(tempdir)))
                with open(os.path.join(tempdir, temporary_file)) as f:
                    assert f.read() == '"fred flintstone",35\n"barney rubble",32\n'
            finally:
                con.drop_table("tempdir_sink", force=True)
    finally:
        con.drop_table("source", force=True)


@pytest.mark.parametrize("table_name", ["new_table", None])
def test_read_csv(con, awards_players_schema, csv_source_configs, table_name):
    source_configs = csv_source_configs("awards_players")
    table = con.read_csv(
        path=source_configs["path"],
        schema=awards_players_schema,
        table_name=table_name,
    )
    try:
        if table_name is None:
            table_name = table.get_name()
        assert table_name in con.list_tables()
        assert table.schema() == awards_players_schema
    finally:
        con.drop_table(table_name)
    assert table_name not in con.list_tables()


@pytest.mark.parametrize("table_name", ["new_table", None])
def test_read_parquet(con, data_dir, tmp_path, table_name, functional_alltypes_schema):
    fname = Path("functional_alltypes.parquet")
    fname = Path(data_dir) / "parquet" / fname.name
    table = con.read_parquet(
        path=tmp_path / fname.name,
        schema=functional_alltypes_schema,
        table_name=table_name,
    )

    try:
        if table_name is None:
            table_name = table.get_name()
        assert table_name in con.list_tables()
        assert table.schema() == functional_alltypes_schema
    finally:
        con.drop_table(table_name)
    assert table_name not in con.list_tables()


@pytest.mark.parametrize("table_name", ["new_table", None])
def test_read_json(con, data_dir, tmp_path, table_name, functional_alltypes_schema):
    pq = pytest.importorskip("pyarrow.parquet")

    pq_table = pq.read_table(
        data_dir.joinpath("parquet", "functional_alltypes.parquet")
    )
    df = pq_table.to_pandas()

    path = tmp_path / "functional_alltypes.json"
    df.to_json(path, orient="records", lines=True, date_format="iso")
    table = con.read_json(
        path=path, schema=functional_alltypes_schema, table_name=table_name
    )

    try:
        if table_name is None:
            table_name = table.get_name()
        assert table_name in con.list_tables()
        assert table.schema() == functional_alltypes_schema
        assert table.count().execute() == len(pq_table)
    finally:
        con.drop_table(table_name)
    assert table_name not in con.list_tables()


@pytest.mark.parametrize(
    "table_name", ["astronauts", "awards_players", "diamonds", "functional_alltypes"]
)
def test_to_csv(con, tmp_path, table_name):
    table = con.table(table_name)
    out_path = tmp_path / "out.csv"
    con.to_csv(table, out_path)

    source_df = table.to_pandas()
    out_df = pd.read_csv(out_path)

    assert source_df.shape == out_df.shape


@pytest.mark.parametrize(
    "table_name", ["astronauts", "awards_players", "diamonds", "functional_alltypes"]
)
def test_to_parquet(con, tmp_path, table_name):
    table = con.table(table_name)
    out_path = tmp_path / "out.parquet"
    con.to_parquet(table, out_path)

    source_df = table.to_pandas()
    out_df = pd.read_parquet(out_path)

    tm.assert_frame_equal(source_df, out_df)
