from __future__ import annotations

import pytest

import ibis
from ibis.backends.trino.tests.conftest import TestConf as TrinoTestConf


@pytest.fixture(scope="module")
def con(data_dir, tmp_path_factory, worker_id):
    with TrinoTestConf(
        data_dir=data_dir, tmpdir=tmp_path_factory, worker_id=worker_id
    ) as backend:
        yield backend.connection


@pytest.mark.parametrize("dialect", ["trino", "athena"])
@pytest.mark.parametrize(
    ("values", "index", "expected"),
    [
        pytest.param([10, 20, 30], -3, 10, id="negative-first"),
        pytest.param([10, 20, 30], -2, 20, id="negative-middle"),
        pytest.param([10, 20, 30], -1, 30, id="negative-last"),
        pytest.param([10, 20, 30], 0, 10, id="first"),
        pytest.param([10, 20, 30], 1, 20, id="middle"),
        pytest.param([10, 20, 30], 2, 30, id="last"),
        pytest.param([10, 20, 30], -4, None, id="before-start"),
        pytest.param([10, 20, 30], 3, None, id="after-end"),
        pytest.param([], -1, None, id="empty-negative"),
        pytest.param([], 0, None, id="empty-nonnegative"),
        pytest.param(None, -1, None, id="null-array"),
        pytest.param([10, None, 30], -2, None, id="null-element"),
        pytest.param([10, 20, 30], None, None, id="null-index"),
    ],
)
def test_array_index(con, dialect, values, index, expected):
    expr = ibis.literal(values, type="array<int64>")[ibis.literal(index, type="int64")]

    with con.begin() as cursor:
        cursor.execute(ibis.to_sql(expr, dialect=dialect))
        assert cursor.fetchone()[0] == expected


@pytest.mark.parametrize("dialect", ["trino", "athena"])
def test_split_array_negative_index(con, dialect):
    t = con.sql("SELECT 'left:middle:right' AS s")
    expr = t.s.split(":")[-1]

    with con.begin() as cursor:
        cursor.execute(ibis.to_sql(expr, dialect=dialect))
        assert cursor.fetchone()[0] == "right"


@pytest.mark.parametrize("dialect", ["trino", "athena"])
def test_array_index_column(con, dialect):
    t = con.sql(
        """
        SELECT * FROM (
            VALUES
                (1, ARRAY[10, 20, 30], -3),
                (2, ARRAY[10, 20, 30], -2),
                (3, ARRAY[10, 20, 30], -1),
                (4, ARRAY[10, 20, 30], 0),
                (5, ARRAY[10, 20, 30], 1),
                (6, ARRAY[10, 20, 30], 2),
                (7, ARRAY[10, 20, 30], -4),
                (8, ARRAY[10, 20, 30], 3),
                (9, CAST(ARRAY[] AS ARRAY(BIGINT)), -1),
                (10, CAST(NULL AS ARRAY(BIGINT)), -1),
                (11, ARRAY[10, NULL, 30], -2),
                (12, ARRAY[10, 20, 30], CAST(NULL AS INTEGER))
        ) AS t(id, a, idx)
        """
    )
    expr = t.select("id", result=t.a[t.idx]).order_by("id")
    expected = [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 10),
        (5, 20),
        (6, 30),
        (7, None),
        (8, None),
        (9, None),
        (10, None),
        (11, None),
        (12, None),
    ]

    with con.begin() as cursor:
        cursor.execute(ibis.to_sql(expr, dialect=dialect))
        assert list(map(tuple, cursor.fetchall())) == expected
