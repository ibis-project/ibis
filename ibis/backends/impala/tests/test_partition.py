from __future__ import annotations

import os
from posixpath import join as pjoin

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis import util
from ibis.tests.util import assert_equal

pytest.importorskip("impala")

from ibis.backends.impala.compat import ImpylaError  # noqa: E402


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "year": [2009] * 3 + [2010] * 3,
            "month": list(map(str, [1, 2, 3] * 2)),
            "value": list(range(1, 7)),
        },
        index=list(range(6)),
    )
    df = pd.concat([df] * 10, ignore_index=True)
    df["id"] = df.index.values
    return df


@pytest.fixture
def unpart_t(con, df, tmp_db):
    pd_name = f"__ibis_test_partition_{util.guid()}"
    con.create_table(pd_name, df, database=tmp_db)
    assert pd_name in con.list_tables(database=tmp_db), pd_name
    yield con.table(pd_name, database=tmp_db)
    con.drop_table(pd_name, database=tmp_db)


def test_is_partitioned(con, temp_table):
    schema = ibis.schema([("foo", "string"), ("year", "int32"), ("month", "string")])
    con.create_table(temp_table, schema=schema, partition=["year", "month"])
    assert con.table(temp_table).is_partitioned


def test_create_table_with_partition_column(con, temp_table_db):
    schema = ibis.schema(
        [
            ("year", "int32"),
            ("month", "string"),
            ("day", "int8"),
            ("value", "double"),
        ]
    )

    tmp_db, name = temp_table_db
    con.create_table(name, schema=schema, database=tmp_db, partition=["year", "month"])

    # the partition column get put at the end of the table
    ex_schema = ibis.schema(
        [
            ("day", "int8"),
            ("value", "double"),
            ("year", "int32"),
            ("month", "string"),
        ]
    )
    table_schema = con.get_schema(name, database=tmp_db)
    assert_equal(table_schema, ex_schema)

    partition_schema = con.database(tmp_db).table(name).partition_schema()

    expected = ibis.schema([("year", "int32"), ("month", "string")])
    assert_equal(partition_schema, expected)


def test_create_partitioned_separate_schema(con, temp_table):
    schema = ibis.schema([("day", "int8"), ("value", "double")])
    part_schema = ibis.schema([("year", "int32"), ("month", "string")])

    con.create_table(temp_table, schema=schema, partition=part_schema)

    # the partition column get put at the end of the table
    ex_schema = ibis.schema(
        [
            ("day", "int8"),
            ("value", "double"),
            ("year", "int32"),
            ("month", "string"),
        ]
    )
    table_schema = con.get_schema(temp_table)
    assert_equal(table_schema, ex_schema)

    partition_schema = con.table(temp_table).partition_schema()
    assert_equal(partition_schema, part_schema)


def test_unpartitioned_table_get_schema(con):
    tname = "functional_alltypes"
    with pytest.raises(ImpylaError):
        con.table(tname).partition_schema()


def test_insert_select_partitioned_table(con, df, temp_table, unpart_t):
    part_keys = ["year", "month"]

    con.create_table(temp_table, schema=unpart_t.schema(), partition=part_keys)
    part_t = con.table(temp_table)
    unique_keys = df[part_keys].drop_duplicates()

    for i, (year, month) in enumerate(unique_keys.itertuples(index=False)):
        select_stmt = unpart_t[(unpart_t.year == year) & (unpart_t.month == month)]

        # test both styles of insert
        if i:
            part = {"year": year, "month": month}
        else:
            part = [year, month]
        part_t.insert(select_stmt, partition=part)

    verify_partitioned_table(part_t, df, unique_keys)


@pytest.fixture
def tmp_parted(con):
    name = f"tmppart_{util.guid()}"
    yield name
    con.drop_table(name, force=True)


def test_create_partitioned_table_from_expr(con, alltypes, tmp_parted):
    t = alltypes
    expr = t[t.id <= 10][["id", "double_col", "month", "year"]]
    name = tmp_parted
    con.create_table(name, expr, partition=[t.year])
    new = con.table(name)
    expected = expr.execute().sort_values("id").reset_index(drop=True)
    result = new.execute().sort_values("id").reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_add_drop_partition_no_location(con, temp_table):
    schema = ibis.schema([("foo", "string"), ("year", "int32"), ("month", "int16")])
    con.create_table(temp_table, schema=schema, partition=["year", "month"])
    table = con.table(temp_table)

    part = {"year": 2007, "month": 4}

    table.add_partition(part)

    assert len(table.partitions()) == 2

    table.drop_partition(part)

    assert len(table.partitions()) == 1


@pytest.mark.hdfs
def test_add_drop_partition_owned_by_impala(hdfs, con, temp_table):
    schema = ibis.schema([("foo", "string"), ("year", "int32"), ("month", "int16")])
    con.create_table(temp_table, schema=schema, partition=["year", "month"])

    table = con.table(temp_table)

    part = {"year": 2007, "month": 4}

    subdir = util.guid()
    basename = util.guid()
    path = f"/tmp/{subdir}/{basename}"

    parent = os.path.dirname(path)
    hdfs.mkdir(parent, create_parents=True)
    hdfs.chown(parent, owner="impala", group="supergroup")

    table.add_partition(part, location=path)

    assert len(table.partitions()) == 2

    table.drop_partition(part)

    assert len(table.partitions()) == 1


def test_add_drop_partition_hive_bug(con, temp_table):
    schema = ibis.schema([("foo", "string"), ("year", "int32"), ("month", "int16")])
    con.create_table(temp_table, schema=schema, partition=["year", "month"])

    table = con.table(temp_table)

    part = {"year": 2007, "month": 4}

    path = f"/tmp/{util.guid()}"

    table.add_partition(part, location=path)

    assert len(table.partitions()) == 2

    table.drop_partition(part)

    assert len(table.partitions()) == 1


@pytest.mark.hdfs
def test_load_data_partition(con, hdfs, tmp_dir, unpart_t, df, temp_table):
    part_keys = ["year", "month"]

    con.create_table(temp_table, schema=unpart_t.schema(), partition=part_keys)
    part_t = con.table(temp_table)

    # trim the runtime of this test
    df = df[df.month == "1"].reset_index(drop=True)

    unique_keys = df[part_keys].drop_duplicates()

    hdfs_dir = pjoin(tmp_dir, "load-data-partition")

    df2 = df.drop(["year", "month"], axis="columns")

    csv_props = {"serialization.format": ",", "field.delim": ","}

    for i, (year, month) in enumerate(unique_keys.itertuples(index=False)):
        chunk = df2[(df.year == year) & (df.month == month)]
        chunk_path = pjoin(hdfs_dir, f"{i}.csv")

        con.write_dataframe(chunk, chunk_path)

        # test both styles of insert
        if i:
            part = {"year": year, "month": month}
        else:
            part = [year, month]

        part_t.add_partition(part)
        part_t.alter_partition(part, format="text", serde_properties=csv_props)
        part_t.load_data(chunk_path, partition=part)

    hdfs.rm(hdfs_dir, recursive=True)
    verify_partitioned_table(part_t, df, unique_keys)


def verify_partitioned_table(part_t, df, unique_keys):
    result = part_t.execute().sort_values(by="id").reset_index(drop=True)[df.columns]

    tm.assert_frame_equal(result, df)

    parts = part_t.partitions()

    # allow for the total line
    assert len(parts) == len(unique_keys) + 1
