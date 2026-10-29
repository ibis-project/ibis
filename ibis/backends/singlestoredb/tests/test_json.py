"""Tests for core map and array operations on JSON and VECTOR values."""

from __future__ import annotations

import json

import pytest
from pytest import param

import ibis
from ibis.util import gen_name

DOCS = [
    {"a": {"x": 1, "y": [1, 2]}, "k": 5, "s": "str"},
    [10, "x", 30],
    "scalar",
    None,
    {},
]


@pytest.fixture
def docs(con):
    name = gen_name("json_docs")
    with con.begin() as c:
        c.execute(f"CREATE TABLE {name} (id INT, doc JSON)")
        values = ", ".join(
            f"({i}, {'NULL' if doc is None else repr(json.dumps(doc))})"
            for i, doc in enumerate(DOCS)
        )
        c.execute(f"INSERT INTO {name} VALUES {values}")
    yield con.table(name)
    con.drop_table(name, force=True)


def _column(con, t, expr):
    result = con.execute(t.select("id", r=expr).order_by("id")).r
    return [None if value != value else value for value in result]  # NaN -> None


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(
            lambda d: d.map,
            [DOCS[0], None, None, None, {}],
            id="map",
        ),
        param(
            lambda d: d.array,
            [None, [10, "x", 30], None, None, None],
            id="array",
        ),
        param(
            lambda d: d.map.keys(),
            [["a", "k", "s"], None, None, None, []],
            id="map_keys",
        ),
        param(
            lambda d: d["a"].map.keys(),
            [["x", "y"], None, None, None, None],
            id="nested_map_keys",
        ),
        param(
            lambda d: d.map.keys().length(),
            [3, None, None, None, 0],
            id="map_length",
        ),
        param(
            lambda d: d.array.length(),
            [None, 3, None, None, None],
            id="array_length",
        ),
        param(
            lambda d: d.map.contains("k"),
            [True, None, None, None, False],
            id="map_contains",
        ),
        param(
            lambda d: d.map.keys().contains("k"),
            [True, None, None, None, False],
            id="map_keys_contains",
        ),
        param(
            lambda d: d.map.get("s"),
            ["str", None, None, None, None],
            id="map_get",
        ),
        param(
            lambda d: d.map.get("nope", ibis.literal("[]").cast("json")),
            [[], None, None, None, []],
            id="map_get_default",
        ),
        param(
            lambda d: d.array[1],
            [None, "x", None, None, None],
            id="array_index",
        ),
        param(
            lambda d: d.array[-1],
            [None, 30, None, None, None],
            id="array_negative_index",
        ),
        param(
            lambda d: d.array.contains("x"),
            [None, True, None, None, None],
            id="array_contains_string",
        ),
        param(
            lambda d: d.array.contains(30),
            [None, True, None, None, None],
            id="array_contains_number",
        ),
    ],
)
def test_json_map_array_ops(con, docs, make_expr, expected):
    assert _column(con, docs, make_expr(docs.doc)) == expected


VECTORS = [[3, 1.5, 2], None]


@pytest.fixture
def vectors(con):
    name = gen_name("vectors")
    with con.begin() as c:
        c.execute(f"CREATE TABLE {name} (id INT, v VECTOR(3), vi VECTOR(3, I8))")
        values = ", ".join(
            f"({i}, {'NULL' if v is None else repr(json.dumps(v))}, "
            f"{'NULL' if v is None else repr(json.dumps([1, 2, 3]))})"
            for i, v in enumerate(VECTORS)
        )
        c.execute(f"INSERT INTO {name} VALUES {values}")
    yield con.table(name)
    con.drop_table(name, force=True)


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        param(lambda t: t.v.length(), [3, None], id="length"),
        param(lambda t: t.v[0], [3, None], id="index"),
        param(lambda t: t.v[-1], [2, None], id="negative_index"),
        param(lambda t: t.v[3], [None, None], id="index_out_of_range"),
        param(lambda t: t.v.contains(1.5), [True, None], id="contains"),
        param(lambda t: t.v.contains(9), [False, None], id="not_contains"),
        param(lambda t: t.vi[1], [2, None], id="integer_elements"),
    ],
)
def test_vector_array_ops(con, vectors, make_expr, expected):
    """JSON functions convert VECTOR values to JSON arrays."""
    assert _column(con, vectors, make_expr(vectors)) == expected
