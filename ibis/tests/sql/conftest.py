from __future__ import annotations

import pytest

import ibis
from ibis.backends.base.sql.compiler import Compiler, QueryContext
from ibis.tests.expr.mocks import MockBackend


@pytest.fixture(scope="module")
def con():
    return MockBackend()


@pytest.fixture(scope="module")
def alltypes(con):
    return con.table("alltypes")


@pytest.fixture(scope="module")
def functional_alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="module")
def star1(con):
    return con.table("star1")


@pytest.fixture(scope="module")
def star2(con):
    return con.table("star2")


@pytest.fixture(scope="module")
def star3(con):
    return con.table("star3")


@pytest.fixture(scope="module")
def nation(con):
    return con.table("tpch_nation")


@pytest.fixture(scope="module")
def region(con):
    return con.table("tpch_region")


@pytest.fixture(scope="module")
def customer(con):
    return con.table("tpch_customer")


@pytest.fixture(scope="module")
def airlines(con):
    return con.table("airlines")


@pytest.fixture(scope="module")
def foo_t(con):
    return con.table("foo_t")


@pytest.fixture(scope="module")
def bar_t(con):
    return con.table("bar_t")


def get_query(expr):
    ast = Compiler.to_ast(expr, QueryContext(compiler=Compiler))
    return ast.queries[0]


def to_sql(expr, *args, **kwargs) -> str:
    return get_query(expr).compile(*args, **kwargs)


@pytest.fixture(scope="module")
def foo(con):
    return con.table("foo")


@pytest.fixture(scope="module")
def bar(con):
    return con.table("bar")


@pytest.fixture(scope="module")
def t1(con):
    return con.table("t1")


@pytest.fixture(scope="module")
def t2(con):
    return con.table("t2")


@pytest.fixture(scope="module")
def where_uncorrelated_subquery(foo, bar):
    return foo[foo.job.isin(bar.job)]


@pytest.fixture(scope="module")
def not_exists(foo_t, bar_t):
    return foo_t[-(foo_t.key1 == bar_t.key1).any()]


@pytest.fixture(scope="module")
def union(con):
    table = con.table("functional_alltypes")

    t1 = table[table.int_col > 0][
        table.string_col.name("key"),
        table.float_col.cast("double").name("value"),
    ]
    t2 = table[table.int_col <= 0][
        table.string_col.name("key"), table.double_col.name("value")
    ]

    return t1.union(t2, distinct=True)


@pytest.fixture(scope="module")
def union_all(con):
    table = con.table("functional_alltypes")

    t1 = table[table.int_col > 0][
        table.string_col.name("key"),
        table.float_col.cast("double").name("value"),
    ]
    t2 = table[table.int_col <= 0][
        table.string_col.name("key"), table.double_col.name("value")
    ]

    return t1.union(t2, distinct=False)


@pytest.fixture(scope="module")
def intersect(con):
    table = con.table("functional_alltypes")

    t1 = table[table.int_col > 0][
        table.string_col.name("key"),
        table.float_col.cast("double").name("value"),
    ]
    t2 = table[table.int_col <= 0][
        table.string_col.name("key"), table.double_col.name("value")
    ]

    return t1.intersect(t2)


@pytest.fixture(scope="module")
def difference(con):
    table = con.table("functional_alltypes")

    t1 = table[table.int_col > 0][
        table.string_col.name("key"),
        table.float_col.cast("double").name("value"),
    ]
    t2 = table[table.int_col <= 0][
        table.string_col.name("key"), table.double_col.name("value")
    ]

    return t1.difference(t2)


@pytest.fixture(scope="module")
def simple_case(con):
    t = con.table("alltypes")
    return t.g.case().when("foo", "bar").when("baz", "qux").else_("default").end()


@pytest.fixture(scope="module")
def search_case(con):
    t = con.table("alltypes")
    return ibis.case().when(t.f > 0, t.d * 2).when(t.c < 0, t.a * 2).end()


@pytest.fixture(scope="module")
def projection_fuse_filter():
    # Probably test this during the evaluation phase. In SQL, "fusable"
    # table operations will be combined together into a single select
    # statement
    #
    # see ibis #71 for more on this

    t = ibis.table(
        [
            ("a", "int8"),
            ("b", "int16"),
            ("c", "int32"),
            ("d", "int64"),
            ("e", "float32"),
            ("f", "float64"),
            ("g", "string"),
            ("h", "boolean"),
        ],
        "foo",
    )

    proj = t["a", "b", "c"]

    # Rewrite a little more aggressively here
    expr1 = proj[t.a > 0]

    # at one point these yielded different results
    filtered = t[t.a > 0]

    expr2 = filtered[t.a, t.b, t.c]
    expr3 = filtered.select(["a", "b", "c"])

    return expr1, expr2, expr3


@pytest.fixture(scope="module")
def startswith(star1):
    t1 = star1
    return t1.foo_id.startswith("foo")


@pytest.fixture(scope="module")
def endswith(star1):
    t1 = star1
    return t1.foo_id.endswith("foo")
