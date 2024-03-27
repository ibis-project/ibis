"""Tests in this module tests if
(1) Ibis generates the correct SQL for time travel,
(2) The generated SQL is executed by Flink without errors.
They do NOT compare the time travel results against the expected results.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import pandas as pd
import pytest

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.selectors import all

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
            "string_col": dt.string,
            "timestamp_col": dt.timestamp(scale=3),
        }
    )

    df = pd.read_parquet(f"{data_dir}/parquet/functional_alltypes.parquet")
    df = df[list(schema.names)]
    df = df.head(100)
    # print(f"df= \n{df.to_string()}")

    temp_path = tmp_path_factory.mktemp(table_name)
    tbl_properties = tempdir_sink_configs(temp_path)

    # Note: Paimon catalog supports 'warehouse'='file:...' only for temporary tables.
    table = con.create_table(
        table_name,
        schema=schema,
        tbl_properties=tbl_properties,
        watermark=ibis.watermark(
            time_col="timestamp_col", allowed_delay=ibis.interval(seconds=15)
        ),
        temp=True,
    )
    con.insert(
        table_name,
        obj=df,
        schema=schema,
    ).wait()

    return table


@pytest.fixture(scope="module")
def table(con, data_dir, tempdir_sink_configs, tmp_path_factory) -> ir.Table:
    table_name = "table"

    yield create_temp_table(
        table_name=table_name,
        con=con,
        data_dir=data_dir,
        tempdir_sink_configs=tempdir_sink_configs,
        tmp_path_factory=tmp_path_factory,
    )

    con.drop_table(
        name=table_name, temp=True, force=True
    )


@pytest.mark.skip()
def test_match_recognize(con, table):
    """Testing board, will be removed before merging.
    """

    # expr = (
    #     table
    #     .time_travel(timestamp)
    #     .select(all())
    # )

    # sql = con.compile(expr)
    # print(f"sql= \n{sql}")

    con.raw_sql("ADD JAR '/Users/mehmet/Downloads/flink-cep-1.15.0.jar'")

    # sql = """
    # SELECT *
    # FROM {0}
    #     MATCH_RECOGNIZE (
    #         PARTITION BY id
    #         ORDER BY timestamp_col
    #     ) MR;
    # """.format(table.get_name())

#     sql = """SELECT *
# FROM `{0}`
#     MATCH_RECOGNIZE (
#         PARTITION BY `id`
#         ORDER BY `timestamp_col`
#         MEASURES
#             START_ROW.timestamp_col AS start_timestamp,
#             LAST(PRICE_DOWN.timestamp_col) AS bottom_timestamp,
#             LAST(PRICE_UP.timestamp_col) AS end_timestamp
#         ONE ROW PER MATCH
#         AFTER MATCH SKIP TO LAST PRICE_UP
#         PATTERN (START_ROW PRICE_DOWN+ PRICE_UP)
#         DEFINE
#             PRICE_DOWN AS
#                 (LAST(PRICE_DOWN.int_col, 1) IS NULL AND PRICE_DOWN.int_col < START_ROW.int_col) OR
#                     PRICE_DOWN.int_col < LAST(PRICE_DOWN.int_col, 1),
#             PRICE_UP AS
#                 PRICE_UP.int_col > LAST(PRICE_DOWN.int_col, 1)
#     ) MR;
#     """.format(table.get_name())

    sql = """SELECT `T`.`start_timestamp`, `T`.`bottom_timestamp`, `T`.`end_timestamp`
FROM `{0}`
    MATCH_RECOGNIZE (
        PARTITION BY `id`, `bool_col`
        ORDER BY `timestamp_col`
        MEASURES
            START_ROW.timestamp_col AS start_timestamp,
            LAST(PRICE_DOWN.timestamp_col) AS bottom_timestamp,
            LAST(PRICE_UP.timestamp_col) AS end_timestamp
        ONE ROW PER MATCH
        AFTER MATCH SKIP TO LAST PRICE_UP
        PATTERN (START_ROW PRICE_DOWN+ PRICE_UP)
        DEFINE
            PRICE_DOWN AS
                (LAST(PRICE_DOWN.int_col, 1) IS NULL AND PRICE_DOWN.int_col < START_ROW.int_col) OR
                    PRICE_DOWN.int_col < LAST(PRICE_DOWN.int_col, 1),
            PRICE_UP AS
                PRICE_UP.int_col > LAST(PRICE_DOWN.int_col, 1)
    ) as T;
    """.format(table.get_name())

    con.raw_sql(sql)

    return


def test_pattern_variable(table):
    var = table.pattern_variable("var")
    assert isinstance(var, ir.MatchRecognizeVariable)

    definition = var.int_col >= 0
    var = var.define(definition)

    assert var.op().definition == definition.op()

    var = var.quantify(1, reluctant=True)
    quantifier = var.op().quantifier

    assert quantifier.min_num_rows == 1
    assert quantifier.max_num_rows == None
    assert quantifier.reluctant == True

    var = var.quantify(0, 10, reluctant=False)

    quantifier = var.op().quantifier
    assert quantifier.min_num_rows == 0
    assert quantifier.max_num_rows == 10
    assert quantifier.reluctant == False

    definition = ibis.and_(var.int_col >= 0, var.bool_col)
    var = var.define(definition)

    assert var.op().definition == definition.op()


@pytest.fixture(
    params=[
        lambda var_a, var_b: var_a.int_col >= 0,
        lambda var_a, var_b: var_a.int_col.first() >= 0,
        lambda var_a, var_b: var_a.int_col.sum() >= 0,
        lambda var_a, var_b: var_a.int_col.first(2).notnull(),
        lambda var_a, var_b: ibis.and_(var_a.int_col.first() >= 0, var_b.bool_col),
        lambda var_a, var_b: ibis.or_(var_a.int_col.first(2) < 10, var_b.smallint_col > 0),
        lambda var_a, var_b: ibis.and_(var_a.int_col.last() == 1, var_b.int_col.mean() > 0),
        lambda var_a, var_b: ibis.and_(var_a.int_col.max() < 10, var_b.int_col.min() > 0),
    ]
)
def lambda_var_definition(request):
    return request.param


def test_pattern_variable_define(table, lambda_var_definition):
    var_a = table.pattern_variable("a")
    var_b = table.pattern_variable("b")

    definition = lambda_var_definition(var_a, var_b)
    var_a = var_a.define(definition)


@pytest.fixture(
    params=[
        (0, None),
        # (0, 1),
        # (0, 10),
        # (10, None),
        # (10, 50),
    ]
)
def quantifier_min_and_max_num_rows(request):
    return request.param


@pytest.mark.parametrize(
    "reluctant", [False, True]
)
def test_pattern_variable_quantify(table, quantifier_min_and_max_num_rows, reluctant):
    var = table.pattern_variable("var")

    min_num_rows, max_num_rows = quantifier_min_and_max_num_rows
    var.quantify(min_num_rows, max_num_rows, reluctant)


@pytest.mark.skip()
def test_pattern_variable_with_chaining(table):
    # TODO (mehmet): The following fails with
    # E ibis.common.annotations.SignatureValidationError: MatchRecognizeVariable(name='b', table=<ibis.expr.operations.relations.DatabaseTable object at 0x193d5e8c0>, definition=(_.int_col >= 0), quantifier=None) has failed due to the following errors:
    # E   `definition`: (_.int_col >= 0) is not either None or coercible to a Value
    # E
    # E Expected signature: MatchRecognizeVariable(name: str, table: Relation, definition: Optional[Value] = None, quantifier: Optional[Quantifier] = None)
    #
    # Look into how to get this working.
    from ibis import _

    var_b = (
        table
        .pattern_variable("b")
        .define(_.int_col >= 0)
        .quantify(1, reluctant=True)
        .define(ibis.and_(_.int_col >= 0, _.bool_col))
        .quantify(0, 10, reluctant=False)
    )


@pytest.fixture(
    params=[
        lambda var: var.int_col.first(),
        lambda var: var.int_col.first(2),
        lambda var: var.int_col.last(),
        lambda var: var.int_col.last(5),
        lambda var: var.int_col.max(),
        lambda var: var.int_col.mean(),
        lambda var: var.int_col.min(),
        lambda var: var.int_col.sum(),
    ]
)
def lambda_measurement_definition(request):
    return request.param


def test_pattern_measurement(table, lambda_measurement_definition):
    var = table.pattern_variable("a")

    measurement_name = "my_measurement"
    definition = lambda_measurement_definition(var)
    measurement = ibis.pattern_measurement(measurement_name, definition)

    assert isinstance(measurement, ir.MatchRecognizeMeasure)
    assert measurement.op().name == measurement_name


@pytest.mark.parametrize(
    "strategy",
    [
        "skip past last",
        "skip PAST last",
        "skip to next",
        "skip TO next",
        "skip to first",
        "SkiP to fIRst",
        "skip to last",
        "SKIP TO LAST",
    ]
)
def test_pattern_after_match(strategy):
    from ibis.expr.operations.match_recognize import AfterMatchStrategy

    after_match = ibis.pattern_after_match(strategy)
    assert isinstance(after_match, ir.MatchRecognizeAfterMatch)
    assert after_match.op().strategy == AfterMatchStrategy.from_str(strategy)


@pytest.mark.parametrize(
    "strategy",
    [
        "skip pastlast",
        "sskip to next",
        "skip to mid",
    ]
)
def test_pattern_after_match_w_invalid_strategy(strategy):
    with pytest.raises(ValueError):
        ibis.pattern_after_match(strategy)


@pytest.fixture(
    params=[
        lambda table: table.id,
        lambda table: [table.id, table.bool_col],
        lambda table: None
    ]
)
def partition_by(table, request):
    return request.param(table)


@pytest.mark.parametrize(
    "reluctant", [False, True]
)
@pytest.mark.parametrize(
    "after_match_strategy",
    [
        "skip past last",
        "skip to next",
        "skip to first",
        "skip to last",
    ]
)
def test_match_recognize(
    con,
    table,
    lambda_var_definition,
    quantifier_min_and_max_num_rows,
    reluctant,
    lambda_measurement_definition,
    after_match_strategy,
    partition_by,
):
    var_a = table.pattern_variable("a")
    var_b = table.pattern_variable("b")

    definition = lambda_var_definition(var_a, var_b)
    var_a = var_a.define(definition)
    var_b = var_b.define(var_b.bool_col)

    min_num_rows, max_num_rows = quantifier_min_and_max_num_rows
    var_a.quantify(min_num_rows, max_num_rows, reluctant)

    expr = table.match_recognize(
        order_by=[table.timestamp_col, table.int_col],
        variables=[var_a, var_b],
        measures=[
            ibis.pattern_measurement("var_a_measurement", definition),
            ibis.pattern_measurement("var_b_measurement", var_b.smallint_col.mean()),
        ],
        after_match=ibis.pattern_after_match(after_match_strategy, var_b),
        partition_by=partition_by,
    )

    expr = expr.select(all())
    # expr.visualize()

    sql = con.compile(expr)
    print(f"sql= \n{sql}")

    # import pdb; pdb.set_trace()

    df = expr.to_pandas()
    print(f"df= \n{df}")

    return

    va = (
        va
        .define(va.int_col >= 0)
        # .quantify(1)
        .quantify(1, reluctant=True)
    )

    vb = (
        vb
        .define(
            # ibis.and_(va.int_col > vb.int_col, vb.smallint_col >= 0)
            vb.int_col >= 0
        )
        .quantify(0, 1)
    )

    expr = table.match_recognize(
        order_by=[table.timestamp_col, table.int_col],
        variables=[va, vb],
        measures=[
            # ibis.pattern_measurement("a_int_col_sum", va.int_col.sum()),
            ibis.pattern_measurement("a_int_col_mean", va.int_col.mean()),
            ibis.pattern_measurement("b_int_col_sum", vb.int_col.sum()),
            # ibis.pattern_measurement("b_ts", vb.timestamp_col),
        ],
        after_match=ibis.pattern_after_match("skip past last"),
        # after_match=ibis.pattern_after_match("skip to first", va),
        # partition_by=[table.id, table.bool_col],
        partition_by=table.bool_col,
        # partition_by="bool_col",
    )

    expr = expr.select(all())
    # expr.visualize()

    sql = con.compile(expr)
    print(f"sql= \n{sql}")

    # import pdb; pdb.set_trace()

    df = expr.to_pandas()
    print(f"df= \n{df}")
