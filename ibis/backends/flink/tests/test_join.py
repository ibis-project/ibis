from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

import ibis
from ibis.backends.flink.tests.conftest import TestConf as tm
from ibis.backends.tests.errors import Py4JJavaError


@pytest.fixture(scope="module")
def left_tmp():
    return tempfile.NamedTemporaryFile()


@pytest.fixture(scope="module")
def right_tmp():
    return tempfile.NamedTemporaryFile()


@pytest.fixture(scope="module")
def left_tumble(con, left_tmp):
    left_pd = pd.DataFrame(
        {
            "row_time": [
                pd.to_datetime("2020-04-15 12:02"),
                pd.to_datetime("2020-04-15 12:06"),
                pd.to_datetime("2020-04-15 12:03"),
            ],
            "num": [1, 2, 3],
            "id": ["L1", "L2", "L3"],
        }
    )
    left_pd.to_csv(left_tmp.name, header=False, index=None)

    left_schema = ibis.schema(
        {
            "row_time": "timestamp(3)",
            "num": "int32",
            "id": "string",
        }
    )
    left = con.create_table(
        "left",
        schema=left_schema,
        tbl_properties={
            "connector": "filesystem",
            "path": left_tmp.name,
            "format": "csv",
        },
        watermark=ibis.watermark(
            time_col="row_time",
            allowed_delay=ibis.interval(seconds=1),
        ),
    )
    left_tumble = left.window_by(time_col=left.row_time).tumble(
        window_size=ibis.interval(minutes=5)
    )

    return left_tumble


@pytest.fixture(scope="module")
def right_tumble(con, right_tmp):
    right_pd = pd.DataFrame(
        {
            "row_time": [
                pd.to_datetime("2020-04-15 12:01"),
                pd.to_datetime("2020-04-15 12:04"),
                pd.to_datetime("2020-04-15 12:05"),
            ],
            "num": [2, 3, 4],
            "id": ["R2", "R3", "R4"],
        }
    )
    right_pd.to_csv(right_tmp.name, header=False, index=None)

    right_schema = ibis.schema(
        {
            "row_time": "timestamp(3)",
            "num": "int32",
            "id": "string",
        }
    )
    right = con.create_table(
        "right",
        schema=right_schema,
        tbl_properties={
            "connector": "filesystem",
            "path": right_tmp.name,
            "format": "csv",
        },
        watermark=ibis.watermark(
            time_col="row_time",
            allowed_delay=ibis.interval(seconds=1),
        ),
    )
    right_tumble = right.window_by(time_col=right.row_time).tumble(
        window_size=ibis.interval(minutes=5)
    )

    return right_tumble


@pytest.fixture
def left_tumble_wo_ambiguity(left_tumble):
    # This is required in order to avoid `row_time` being an ambiguous
    # reference
    return left_tumble[left_tumble]


@pytest.fixture
def right_tmp_wo_ambiguity(right_tumble):
    # This is required in order to avoid `row_time` being an ambiguous
    # reference
    return right_tumble[right_tumble]


@pytest.fixture(autouse=True, scope="module")
def remove_temp_files(left_tmp, right_tmp):
    yield
    left_tmp.close()
    right_tmp.close()


@pytest.mark.xfail(
    raises=(Py4JJavaError, AssertionError),
    reason="subquery probably uses too much memory/resources, flink complains about network buffers",
    strict=False,
)
def test_outer_join(left_tumble_wo_ambiguity, right_tmp_wo_ambiguity):
    expr = left_tumble_wo_ambiguity.join(
        right_tmp_wo_ambiguity,
        ["num", "window_start", "window_end"],
        how="outer",
        lname="L_{name}",
        rname="R_{name}",
    )
    expr = expr[
        "L_num",
        "L_id",
        "R_num",
        "R_id",
        ibis.coalesce(expr["L_window_start"], expr["R_window_start"]).name(
            "window_start"
        ),
        ibis.coalesce(expr["L_window_end"], expr["R_window_end"]).name("window_end"),
    ]
    result_df = expr.to_pandas()

    expected_df = pd.DataFrame.from_dict(
        {
            "L_num": {0: np.nan, 1: 1.0, 2: 3.0, 3: 2.0, 4: np.nan},
            "L_id": {0: None, 1: "L1", 2: "L3", 3: "L2", 4: None},
            "R_num": {0: 2.0, 1: np.nan, 2: 3.0, 3: np.nan, 4: 4.0},
            "R_id": {0: "R2", 1: None, 2: "R3", 3: None, 4: "R4"},
            "window_start": {
                0: pd.Timestamp("2020-04-15 12:00:00"),
                1: pd.Timestamp("2020-04-15 12:00:00"),
                2: pd.Timestamp("2020-04-15 12:00:00"),
                3: pd.Timestamp("2020-04-15 12:05:00"),
                4: pd.Timestamp("2020-04-15 12:05:00"),
            },
            "window_end": {
                0: pd.Timestamp("2020-04-15 12:05:00"),
                1: pd.Timestamp("2020-04-15 12:05:00"),
                2: pd.Timestamp("2020-04-15 12:05:00"),
                3: pd.Timestamp("2020-04-15 12:10:00"),
                4: pd.Timestamp("2020-04-15 12:10:00"),
            },
        }
    )
    tm.assert_frame_equal(result_df, expected_df)


@pytest.fixture(
    params=[
        (
            ["num"],
            """SELECT
  `t4`.`num`,
  `t4`.`id`,
  `t4`.`window_start`,
  `t4`.`window_end`
FROM (
  SELECT
    `t0`.*
  FROM TABLE(TUMBLE(TABLE `left`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t0`
) AS `t4` WHERE NOT EXISTS (
  SELECT
    *
  FROM (
    SELECT
      `t1`.*
    FROM TABLE(TUMBLE(TABLE `right`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t1`
  ) AS `t5`
    WHERE `t4`.`num` = `t5`.`num`
)""",
            pd.DataFrame.from_dict(
                {
                    "num": {0: 1},
                    "id": {0: "L1"},
                    "window_start": {
                        0: pd.Timestamp("2020-04-15 12:00:00"),
                    },
                    "window_end": {
                        0: pd.Timestamp("2020-04-15 12:05:00"),
                    },
                }
            ),
        ),
        (
            ["num", "window_start", "window_end"],
            """SELECT
  `t4`.`num`,
  `t4`.`id`,
  `t4`.`window_start`,
  `t4`.`window_end`
FROM (
  SELECT
    `t0`.*
  FROM TABLE(TUMBLE(TABLE `left`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t0`
) AS `t4` WHERE NOT EXISTS (
  SELECT
    *
  FROM (
    SELECT
      `t1`.*
    FROM TABLE(TUMBLE(TABLE `right`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t1`
  ) AS `t5`
    WHERE `t4`.`num` = `t5`.`num`
    AND `t4`.`window_start` = `t5`.`window_start`
    AND `t4`.`window_end` = `t5`.`window_end`
)""",
            pd.DataFrame.from_dict(
                {
                    "num": {0: 1, 1: 2},
                    "id": {0: "L1", 1: "L2"},
                    "window_start": {
                        0: pd.Timestamp("2020-04-15 12:00:00"),
                        1: pd.Timestamp("2020-04-15 12:05:00"),
                    },
                    "window_end": {
                        0: pd.Timestamp("2020-04-15 12:05:00"),
                        1: pd.Timestamp("2020-04-15 12:10:00"),
                    },
                }
            ),
        ),
    ]
)
def anti_join_predicates_and_sql_and_df(request):
    return request.param


@pytest.fixture
def anti_join_expr_and_sql_and_df(
    anti_join_predicates_and_sql_and_df, left_tumble, right_tumble
):
    predicates, expected_sql, expected_df = anti_join_predicates_and_sql_and_df

    expr = left_tumble.join(
        right_tumble,
        predicates=predicates,
        how="anti_window",
    )
    expr = expr[
        "num",
        "id",
        "window_start",
        "window_end",
    ]

    return expr, expected_sql, expected_df


def test_anti_join_sql(anti_join_expr_and_sql_and_df):
    expr, expected_sql, _ = anti_join_expr_and_sql_and_df
    sql = ibis.to_sql(expr, dialect="flink")
    assert sql == expected_sql


def test_anti_join_result(anti_join_expr_and_sql_and_df):
    expr, _, expected_df = anti_join_expr_and_sql_and_df
    result_df = expr.to_pandas()
    tm.assert_frame_equal(result_df, expected_df, check_dtype=False)


@pytest.fixture(
    params=[
        (
            ["num"],
            """SELECT
  `t4`.`num`,
  `t4`.`id`,
  `t4`.`window_start`,
  `t4`.`window_end`
FROM (
  SELECT
    `t0`.*
  FROM TABLE(TUMBLE(TABLE `left`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t0`
) AS `t4` WHERE EXISTS (
  SELECT
    *
  FROM (
    SELECT
      `t1`.*
    FROM TABLE(TUMBLE(TABLE `right`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t1`
  ) AS `t5`
    WHERE `t4`.`num` = `t5`.`num`
)""",
            pd.DataFrame.from_dict(
                {
                    "num": {0: 2, 1: 3},
                    "id": {0: "L2", 1: "L3"},
                    "window_start": {
                        0: pd.Timestamp("2020-04-15 12:05:00"),
                        1: pd.Timestamp("2020-04-15 12:00:00"),
                    },
                    "window_end": {
                        0: pd.Timestamp("2020-04-15 12:10:00"),
                        1: pd.Timestamp("2020-04-15 12:05:00"),
                    },
                }
            ),
        ),
        (
            ["num", "window_start", "window_end"],
            """SELECT
  `t4`.`num`,
  `t4`.`id`,
  `t4`.`window_start`,
  `t4`.`window_end`
FROM (
  SELECT
    `t0`.*
  FROM TABLE(TUMBLE(TABLE `left`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t0`
) AS `t4` WHERE EXISTS (
  SELECT
    *
  FROM (
    SELECT
      `t1`.*
    FROM TABLE(TUMBLE(TABLE `right`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t1`
  ) AS `t5`
    WHERE `t4`.`num` = `t5`.`num`
    AND `t4`.`window_start` = `t5`.`window_start`
    AND `t4`.`window_end` = `t5`.`window_end`
)""",
            pd.DataFrame.from_dict(
                {
                    "num": {0: 3},
                    "id": {0: "L3"},
                    "window_start": {
                        0: pd.Timestamp("2020-04-15 12:00:00"),
                    },
                    "window_end": {
                        0: pd.Timestamp("2020-04-15 12:05:00"),
                    },
                }
            ),
        ),
    ]
)
def semi_join_predicates_and_sql_and_df(request):
    return request.param


@pytest.fixture
def semi_join_expr_and_sql_and_df(
    semi_join_predicates_and_sql_and_df, left_tumble, right_tumble
):
    predicates, expected_sql, expected_df = semi_join_predicates_and_sql_and_df

    expr = left_tumble.join(
        right_tumble,
        predicates=predicates,
        how="semi_window",
    )
    expr = expr[
        "num",
        "id",
        "window_start",
        "window_end",
    ]

    return expr, expected_sql, expected_df


def test_semi_join_sql(semi_join_expr_and_sql_and_df):
    expr, expected_sql, _ = semi_join_expr_and_sql_and_df
    sql = ibis.to_sql(expr, dialect="flink")
    assert sql == expected_sql


def test_semi_join_result(semi_join_expr_and_sql_and_df):
    expr, _, expected_df = semi_join_expr_and_sql_and_df
    result_df = expr.to_pandas()
    tm.assert_frame_equal(result_df, expected_df, check_dtype=False)
