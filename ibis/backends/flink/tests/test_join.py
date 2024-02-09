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
    left_tumble = left_tumble[
        left_tumble
    ]  # this is required in order to avoid `row_time` being an ambiguous reference
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
    right_tumble = right_tumble[
        right_tumble
    ]  # this is required in order to avoid `row_time` being an ambiguous reference
    return right_tumble


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
def test_outer_join(left_tumble, right_tumble):
    expr = left_tumble.join(
        right_tumble,
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
