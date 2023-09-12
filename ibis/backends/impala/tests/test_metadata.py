from __future__ import annotations

import pandas as pd
import pytest
import toolz
from numpy import nan

from ibis.backends.impala.metadata import parse_metadata


@pytest.fixture(scope="module")
def spacer():
    return ("", nan, nan)


@pytest.fixture(scope="module")
def schema(spacer):
    return [
        ("# col_name", "data_type", "comment"),
        spacer,
        ("foo", "int", nan),
        ("bar", "tinyint", nan),
        ("baz", "bigint", nan),
    ]


@pytest.fixture(scope="module")
def partitions(spacer):
    return [
        ("# Partition Information", nan, nan),
        ("# col_name", "data_type", "comment"),
        spacer,
        ("qux", "bigint", nan),
    ]


@pytest.fixture(scope="module")
def info():
    return [
        ("# Detailed Table Information", nan, nan),
        ("Database:", "tpcds", nan),
        ("Owner:", "wesm", nan),
        ("CreateTime:", "2015-11-08 01:09:42-08:00", nan),
        ("LastAccessTime:", "UNKNOWN", nan),
        ("Protect Mode:", "None", nan),
        ("Retention:", "0", nan),
        ("Location:", "hdfs://host-name:20500/my.db/dbname.table_name", nan),
        ("Table Type:", "EXTERNAL_TABLE", nan),
        ("Table Parameters:", nan, nan),
        ("", "EXTERNAL", "TRUE"),
        ("", "STATS_GENERATED_VIA_STATS_TASK", "true"),
        ("", "numRows", "183592"),
        ("", "transient_lastDdlTime", "1447340941"),
    ]


@pytest.fixture(scope="module")
def storage_info():
    return [
        ("# Storage Information", nan, nan),
        ("SerDe Library:", "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe", nan),
        ("InputFormat:", "org.apache.hadoop.mapred.TextInputFormat", nan),
        (
            "OutputFormat:",
            "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            nan,
        ),
        ("Compressed:", "No", nan),
        ("Num Buckets:", "0", nan),
        ("Bucket Columns:", "[]", nan),
        ("Sort Columns:", "[]", nan),
        ("Storage Desc Params:", nan, nan),
        ("", "field.delim", "|"),
        ("", "serialization.format", "|"),
    ]


@pytest.fixture(scope="module")
def part_metadata(spacer, schema, partitions, info, storage_info):
    return pd.DataFrame.from_records(
        list(
            toolz.concat(
                toolz.interpose([spacer], [schema, partitions, info, storage_info])
            )
        ),
        columns=["name", "type", "comment"],
    )


@pytest.fixture(scope="module")
def unpart_metadata(spacer, schema, info, storage_info):
    return pd.DataFrame.from_records(
        list(toolz.concat(toolz.interpose([spacer], [schema, info, storage_info]))),
        columns=["name", "type", "comment"],
    )


@pytest.fixture(scope="module")
def parsed_part(part_metadata):
    return parse_metadata(part_metadata)


@pytest.fixture(scope="module")
def parsed_unpart(unpart_metadata):
    return parse_metadata(unpart_metadata)


def test_table_params(parsed_part):
    params = parsed_part.info["Table Parameters"]

    assert params["EXTERNAL"] is True
    assert params["STATS_GENERATED_VIA_STATS_TASK"] is True
    assert params["numRows"] == 183592
    assert params["transient_lastDdlTime"] == pd.Timestamp("2015-11-12 15:09:01")


def test_partitions(parsed_unpart, parsed_part):
    assert parsed_unpart.partitions is None
    assert parsed_part.partitions == [("qux", "bigint")]


def test_schema(parsed_part):
    assert parsed_part.schema == [
        ("foo", "int"),
        ("bar", "tinyint"),
        ("baz", "bigint"),
    ]


def test_storage_info(parsed_part):
    storage = parsed_part.storage
    assert storage["Compressed"] is False
    assert storage["Num Buckets"] == 0


def test_storage_params(parsed_part):
    params = parsed_part.storage["Desc Params"]

    assert params["field.delim"] == "|"
    assert params["serialization.format"] == "|"
