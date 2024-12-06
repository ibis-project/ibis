from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis import util
from ibis.tests.util import assert_equal

pytest.importorskip("impala")

from impala.error import Error as ImpylaError  # noqa: E402


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
def unpart_t(con, df):
    pd_name = f"__ibis_test_partition_{util.guid()}"
    t = con.create_table(pd_name, df)
    assert pd_name in con.list_tables(), pd_name
    yield t
    con.drop_table(pd_name)


def test_create_table_with_partition_column(con, temp_table):
    schema = ibis.schema(
        [
            ("year", "int32"),
            ("month", "string"),
            ("day", "int8"),
            ("value", "double"),
        ]
    )

    con.create_table(temp_table, schema=schema, partition=["year", "month"])

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

    partition_schema = con.get_partition_schema(temp_table)

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

    partition_schema = con.get_partition_schema(temp_table)
    assert_equal(partition_schema, part_schema)


def test_unpartitioned_table_get_schema(con):
    tname = "functional_alltypes"
    with pytest.raises(ImpylaError):
        con.get_partition_schema(tname)


def test_insert_select_partitioned_table(con, df, temp_table, unpart_t):
    part_keys = ["year", "month"]

    con.create_table(temp_table, schema=unpart_t.schema(), partition=part_keys)
    part_t = con.table(temp_table)
    unique_keys = df[part_keys].drop_duplicates()

    for i, (year, month) in enumerate(unique_keys.itertuples(index=False)):
        select_stmt = unpart_t.filter(
            (unpart_t.year == year) & (unpart_t.month == month)
        )

        # test both styles of insert
        if i:
            part = {"year": year, "month": month}
        else:
            part = [year, month]
        con.insert(temp_table, select_stmt, partition=part)

    result = part_t.execute().sort_values(by="id").reset_index(drop=True)[df.columns]

    tm.assert_frame_equal(result, df)

    parts = con.list_partitions(temp_table)

    # allow for the total line
    assert len(parts) == len(unique_keys) + 1


@pytest.fixture
def tmp_parted(con):
    name = f"tmppart_{util.guid()}"
    yield name
    con.drop_table(name, force=True)


def test_create_partitioned_table_from_expr(con, alltypes, tmp_parted):
    t = alltypes
    expr = t.filter(t.id <= 10)[["id", "double_col", "month", "year"]]
    name = tmp_parted
    con.create_table(name, expr, partition=[t.year])
    new = con.table(name)
    expected = expr.execute().sort_values("id").reset_index(drop=True)
    result = new.execute().sort_values("id").reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_add_drop_partition_no_location(con, temp_table):
    schema = ibis.schema([("foo", "string"), ("year", "int32"), ("month", "int16")])
    con.create_table(
        temp_table,
        schema=schema,
        partition=["year", "month"],
        tbl_properties={"transactional": "false"},
    )

    part = {"year": 2007, "month": 4}

    con.add_partition(temp_table, part)

    assert len(con.list_partitions(temp_table)) == 2

    con.drop_partition(temp_table, part)

    assert len(con.list_partitions(temp_table)) == 1


def test_add_drop_partition_owned_by_impala(con, temp_table):
    schema = ibis.schema([("foo", "string"), ("year", "int32"), ("month", "int16")])
    con.create_table(
        temp_table,
        schema=schema,
        partition=["year", "month"],
        tbl_properties={"transactional": "false"},
    )
    part = {"year": 2007, "month": 4}

    subdir = util.guid()
    basename = util.guid()
    path = f"/tmp/{subdir}/{basename}"

    con.add_partition(temp_table, part, location=path)

    assert len(con.list_partitions(temp_table)) == 2

    con.drop_partition(temp_table, part)

    assert len(con.list_partitions(temp_table)) == 1


def test_add_drop_partition_hive_bug(con, temp_table):
    schema = ibis.schema([("foo", "string"), ("year", "int32"), ("month", "int16")])
    con.create_table(
        temp_table,
        schema=schema,
        partition=["year", "month"],
        tbl_properties={"transactional": "false"},
    )

    part = {"year": 2007, "month": 4}

    path = f"/tmp/{util.guid()}"

    con.add_partition(temp_table, part, location=path)

    assert len(con.list_partitions(temp_table)) == 2

    con.drop_partition(temp_table, part)

    assert len(con.list_partitions(temp_table)) == 1
