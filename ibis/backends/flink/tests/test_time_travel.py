"""Tests in this module tests if
(1) Ibis generates the correct SQL for time travel,
(2) The generated SQL is executed by Flink without errors.
They do NOT compare the time travel results against the expected results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.flink.tests.utils import download_jar_for_package, get_catalogs

if TYPE_CHECKING:
    from pathlib import Path


def create_temp_table(
    table_name: str,
    con,
    data_dir: Path,
    tempdir_sink_configs,
    tmp_path_factory,
):
    # Subset of `functional_alltypes_schema`
    schema = sch.Schema(
        {
            "id": dt.int32,
            "bool_col": dt.bool,
            "smallint_col": dt.int16,
            "int_col": dt.int32,
            "timestamp_col": dt.timestamp(scale=3),
        }
    )

    df = pd.read_parquet(f"{data_dir}/parquet/functional_alltypes.parquet")
    df = df[list(schema.names)]
    df = df.head(20)

    temp_path = tmp_path_factory.mktemp(table_name)
    tbl_properties = tempdir_sink_configs(temp_path)

    # Note: Paimon catalog supports 'warehouse'='file:...' only for temporary tables.
    table = con.create_table(
        table_name,
        schema=schema,
        tbl_properties=tbl_properties,
        temp=True,
    )
    con.insert(
        table_name,
        obj=df,
        schema=schema,
    ).wait()

    return table


@pytest.fixture(scope="module")
def temp_table(con, data_dir, tempdir_sink_configs, tmp_path_factory) -> ir.Table:
    table_name = "test_table"

    yield create_temp_table(
        table_name=table_name,
        con=con,
        data_dir=data_dir,
        tempdir_sink_configs=tempdir_sink_configs,
        tmp_path_factory=tmp_path_factory,
    )

    con.drop_table(name=table_name, temp=True, force=True)


@pytest.fixture(
    params=[
        (
            ibis.timestamp("2023-01-02T03:04:05"),
            "CAST('2023-01-02 03:04:05.000000' AS TIMESTAMP)",
        ),
        (
            ibis.timestamp("2023-01-01 12:34:56.789 UTC"),
            "CAST('2023-01-01 12:34:56.789000' AS TIMESTAMP)",
        ),
        (
            ibis.timestamp("2023-01-02T03:04:05") + ibis.interval(days=3),
            "CAST('2023-01-02 03:04:05.000000' AS TIMESTAMP) + INTERVAL '3' DAY(2)",
        ),
        (
            ibis.timestamp("2023-01-02T03:04:05")
            + ibis.interval(days=3)
            + ibis.interval(hours=6),
            "(CAST('2023-01-02 03:04:05.000000' AS TIMESTAMP) + INTERVAL '3' DAY(2)) + INTERVAL '6' HOUR(2)",
        ),
        (
            ibis.timestamp("2023-01-02 03:04:05", timezone="EST"),
            "CAST('2023-01-02 03:04:05.000000' AS TIMESTAMP)",
        ),
    ]
)
def timestamp_and_sql(request):
    return request.param


def test_time_travel(temp_table, timestamp_and_sql):
    timestamp, _ = timestamp_and_sql
    expr = temp_table.time_travel(timestamp)

    assert expr.op().timestamp == timestamp.op()


@pytest.fixture
def time_travel_expr_and_expected_sql(
    temp_table, timestamp_and_sql
) -> tuple[ir.Expr, str]:
    from ibis.selectors import all

    timestamp, timestamp_sql = timestamp_and_sql

    expr = temp_table.time_travel(timestamp).select(all())
    expected_sql = f"""SELECT `t0`.`id`, `t0`.`bool_col`, `t0`.`smallint_col`, `t0`.`int_col`, `t0`.`timestamp_col` FROM `{temp_table.get_name()}` FOR SYSTEM_TIME AS OF {timestamp_sql} AS `t0`"""

    return expr, expected_sql


def test_time_travel_compile(con, time_travel_expr_and_expected_sql):
    expr, expected_sql = time_travel_expr_and_expected_sql
    sql = con.compile(expr)
    assert sql == expected_sql


@pytest.fixture
def use_hive_catalog(con):
    # Flink related
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="flink-sql-connector-hive-3.1.3_2.12-1.18.1",
        jar_url="https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-hive-3.1.3_2.12/1.18.1/flink-sql-connector-hive-3.1.3_2.12-1.18.1.jar",
    )

    # Hadoop related
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="woodstox-core-5.3.0",
        jar_url="https://repo1.maven.org/maven2/com/fasterxml/woodstox/woodstox-core/5.3.0/woodstox-core-5.3.0.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="commons-logging-1.1.3",
        jar_url="https://repo1.maven.org/maven2/commons-logging/commons-logging/1.1.3/commons-logging-1.1.3.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="commons-configuration2-2.1.1",
        jar_url="https://repo1.maven.org/maven2/org/apache/commons/commons-configuration2/2.1.1/commons-configuration2-2.1.1.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="hadoop-auth-3.3.2",
        jar_url="https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-auth/3.3.2/hadoop-auth-3.3.2.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="hadoop-common-3.3.2",
        jar_url="https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.2/hadoop-common-3.3.2.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="hadoop-hdfs-client-3.3.2",
        jar_url="https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-hdfs-client/3.3.2/hadoop-hdfs-client-3.3.2.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="hadoop-mapreduce-client-core-3.3.2",
        jar_url="https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-mapreduce-client-core/3.3.2/hadoop-mapreduce-client-core-3.3.2.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="hadoop-shaded-guava-1.1.1",
        jar_url="https://repo1.maven.org/maven2/org/apache/hadoop/thirdparty/hadoop-shaded-guava/1.1.1/hadoop-shaded-guava-1.1.1.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="stax2-api-4.2.1",
        jar_url="https://repo1.maven.org/maven2/org/codehaus/woodstox/stax2-api/4.2.1/stax2-api-4.2.1.jar",
    )

    hive_catalog = "hive_catalog"
    sql = """
    CREATE CATALOG hive_catalog WITH (
        'type' = 'hive',
        'hive-conf-dir' = './docker/flink/conf/'
    );
    """
    con.raw_sql(sql)

    catalog_list = get_catalogs(con)
    assert hive_catalog in catalog_list

    con.raw_sql(f"USE CATALOG {hive_catalog};")


@pytest.fixture
def use_paimon_catalog(con):
    # Note: It is not ideal to do "test ops" in code here. However,
    # adding JAR files in the Flink container in this case won't help
    # because the Flink specific tests do not run on the dockerized env,
    # but on the local env.
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="paimon-flink-1.18-0.8-20240301.002155-30",
        jar_url="https://repository.apache.org/content/groups/snapshots/org/apache/paimon/paimon-flink-1.18/0.8-SNAPSHOT/paimon-flink-1.18-0.8-20240301.002155-30.jar",
    )
    download_jar_for_package(
        package_name="apache-flink",
        jar_name="flink-shaded-hadoop-2-uber-2.8.3-10.0",
        jar_url="https://repo.maven.apache.org/maven2/org/apache/flink/flink-shaded-hadoop-2-uber/2.8.3-10.0/flink-shaded-hadoop-2-uber-2.8.3-10.0.jar",
    )

    paimon_catalog = "paimon_catalog"
    catalog_list = get_catalogs(con)
    if paimon_catalog not in catalog_list:
        sql = """
        CREATE CATALOG paimon_catalog WITH (
            'type'='paimon',
            'warehouse'='file:/tmp/paimon'
        );
        """
        con.raw_sql(sql)
        catalog_list = get_catalogs(con)
        assert paimon_catalog in catalog_list

    con.raw_sql(f"USE CATALOG {paimon_catalog};")


@pytest.fixture
def time_travel_expr(
    con, data_dir, tempdir_sink_configs, tmp_path_factory, timestamp_and_sql
) -> ir.Expr:
    from ibis.selectors import all

    table_name = "test_table"
    table = create_temp_table(
        con=con,
        table_name=table_name,
        data_dir=data_dir,
        tempdir_sink_configs=tempdir_sink_configs,
        tmp_path_factory=tmp_path_factory,
    )

    timestamp, _ = timestamp_and_sql

    yield table.time_travel(timestamp).select(all())

    con.drop_table(name=table_name, temp=True, force=True)


# Note: `test_time_travel_w_xxx_catalog()` tests rely on `use_hive_catalog`
# to appear before `time_travel_expr` per "Fixtures are evaluated in order
# of presence in test function arguments, from left to right."
# This is required to create the table in the catalog.
@pytest.mark.skip(
    reason=(
        "Fixture `use_hive_catalog` uses `download_jar_for_package()`, "
        "which does not work on CI. Downloading the JAR's in Flink container "
        "is also not an option because tests under `ibis/backends/flink/tests/` "
        "run on local env. So this is left to be run manually for now."
    )
)
def test_time_travel_w_hive_catalog(con, use_hive_catalog, time_travel_expr):
    # Start the Hive metastore first
    # $ docker compose up hive-metastore --force-recreate
    con.execute(time_travel_expr)


@pytest.mark.skip(
    reason=(
        "Fixture `use_paimon_catalog` uses `download_jar_for_package()`, "
        "which does not work on CI. Downloading the JAR's in Flink container "
        "is also not an option because tests under `ibis/backends/flink/tests/` "
        "run on local env. So this is left to be run manually for now."
    )
)
def test_time_travel_w_paimon_catalog(con, use_paimon_catalog, time_travel_expr):
    con.execute(time_travel_expr)
