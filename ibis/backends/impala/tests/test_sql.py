from __future__ import annotations

import operator

import pytest

import ibis
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.tests.mocks import MockImpalaConnection


@pytest.fixture(scope="module")
def con():
    return MockImpalaConnection()


@pytest.fixture
def limit_cte_extract(con):
    alltypes = con.table("functional_alltypes")
    t = alltypes.limit(100)
    t2 = t.view()
    return t.join(t2).select(t)


@pytest.mark.parametrize(
    "join_type", ["cross_join", "inner_join", "left_join", "outer_join"]
)
def test_join_no_predicates_for_impala(con, join_type, snapshot):
    t1 = con.table("star1")
    t2 = con.table("star2")

    joined = getattr(t1, join_type)(t2)[[t1]]
    result = ImpalaCompiler.to_sql(joined)
    snapshot.assert_match(result, "out.sql")


def test_limit_cte_extract(limit_cte_extract, snapshot):
    case = limit_cte_extract
    result = ImpalaCompiler.to_sql(case)
    snapshot.assert_match(result, "out.sql")


def test_nested_join_base(snapshot):
    t = ibis.table(dict(uuid="string", ts="timestamp"), name="t")
    counts = t.group_by("uuid").size()
    max_counts = counts.group_by("uuid").aggregate(max_count=lambda x: x[1].max())
    result = max_counts.left_join(counts, "uuid").select(counts)
    compiled_result = ImpalaCompiler.to_sql(result)
    snapshot.assert_match(compiled_result, "out.sql")


def test_nested_joins_single_cte(snapshot):
    t = ibis.table(dict(uuid="string", ts="timestamp"), name="t")

    counts = t.group_by("uuid").size()

    last_visit = t.group_by("uuid").aggregate(last_visit=t.ts.max())

    max_counts = counts.group_by("uuid").aggregate(max_count=counts[1].max())

    main_kw = max_counts.left_join(
        counts, ["uuid", max_counts.max_count == counts[1]]
    ).select(counts)

    result = main_kw.left_join(last_visit, "uuid").select(
        main_kw, last_visit.last_visit
    )
    compiled_result = ImpalaCompiler.to_sql(result)
    snapshot.assert_match(compiled_result, "out.sql")


def test_nested_join_multiple_ctes(snapshot):
    ratings = ibis.table(
        dict(userid="int64", movieid="int64", rating="int8", timestamp="string"),
        name="ratings",
    )
    movies = ibis.table(dict(movieid="int64", title="string"), name="movies")

    expr = ratings.timestamp.cast("timestamp")
    ratings2 = ratings["userid", "movieid", "rating", expr.name("datetime")]
    joined2 = ratings2.join(movies, ["movieid"])[ratings2, movies["title"]]
    joined3 = joined2.filter([joined2.userid == 118205, joined2.datetime.year() > 2001])
    top_user_old_movie_ids = joined3.filter(
        [joined3.userid == 118205, joined3.datetime.year() < 2009]
    )[["movieid"]]
    # projection from a filter was hiding an insidious bug, so we're disabling
    # that for now see issue #1295
    cond = joined3.movieid.isin(top_user_old_movie_ids.movieid)
    result = joined3[cond]
    compiled_result = ImpalaCompiler.to_sql(result)
    snapshot.assert_match(compiled_result, "out.sql")


def test_logically_negate_complex_boolean_expr(snapshot):
    t = ibis.table(
        [("a", "string"), ("b", "double"), ("c", "int64"), ("d", "string")],
        name="t",
    )

    def f(t):
        return t.a.isin(["foo"]) & t.c.notnull()

    expr = (~f(t)).name("tmp")
    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_join_with_nested_or_condition(snapshot):
    t1 = ibis.table([("a", "string"), ("b", "string")], "t")
    t2 = t1.view()

    joined = t1.join(t2, [t1.a == t2.a, (t1.a != t2.b) | (t1.b != t2.a)])
    expr = joined[t1]
    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_join_with_nested_xor_condition(snapshot):
    t1 = ibis.table([("a", "string"), ("b", "string")], "t")
    t2 = t1.view()

    joined = t1.join(t2, [t1.a == t2.a, (t1.a != t2.b) ^ (t1.b != t2.a)])
    expr = joined[t1]
    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("method", ["isnull", "notnull"])
def test_is_parens(method, snapshot):
    t = ibis.table([("a", "string"), ("b", "string")], "table")
    func = operator.methodcaller(method)
    expr = t[func(t.a) == func(t.b)]

    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_is_parens_identical_to(snapshot):
    t = ibis.table([("a", "string"), ("b", "string")], "table")
    expr = t[t.a.identical_to(None) == t.b.identical_to(None)]

    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_join_aliasing(snapshot):
    test = ibis.table(
        [("a", "int64"), ("b", "int64"), ("c", "int64")], name="test_table"
    )
    test = test.mutate(d=test.a + 20)
    test2 = test[test.d, test.c]
    idx = (test2.d / 15).cast("int64").name("idx")
    test3 = test2.group_by([test2.d, idx, test2.c]).aggregate(row_count=test2.count())
    test3_totals = test3.group_by(test3.d).aggregate(total=test3.row_count.sum())
    test4 = test3.join(test3_totals, test3.d == test3_totals.d)[
        test3, test3_totals.total
    ]
    test5 = test4[test4.row_count < test4.total / 2]
    agg = (
        test.group_by([test.d, test.b])
        .aggregate(count=test.count(), unique=test.c.nunique())
        .view()
    )
    result = agg.join(test5, agg.d == test5.d)[agg, test5.total]
    result = ImpalaCompiler.to_sql(result)
    snapshot.assert_match(result, "out.sql")


def test_multiple_filters(snapshot):
    t = ibis.table([("a", "int64"), ("b", "string")], name="t0")
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_multiple_filters2(snapshot):
    t = ibis.table([("a", "int64"), ("b", "string")], name="t0")
    filt = t[t.a < 100]
    expr = filt[filt.a == filt.a.max()]
    expr = expr[expr.b == "a"]
    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.fixture
def region():
    return ibis.table(
        dict(r_regionkey="int16", r_name="string", r_comment="string"),
        name="tpch_region",
    )


@pytest.fixture
def nation():
    return ibis.table(
        dict(
            n_nationkey="int32",
            n_name="string",
            n_regionkey="int32",
            n_comment="string",
        ),
        name="tpch_nation",
    )


@pytest.fixture
def customer():
    return ibis.table(
        dict(
            c_custkey="int64",
            c_name="string",
            c_address="string",
            c_nationkey="int32",
            c_phone="string",
            c_acctbal="decimal(12, 2)",
            c_mktsegment="string",
            c_comment="string",
        ),
        name="tpch_customer",
    )


@pytest.fixture
def orders():
    return ibis.table(
        dict(
            o_orderkey="int64",
            o_custkey="int64",
            o_orderstatus="string",
            o_totalprice="decimal(12, 2)",
            o_orderdate="string",
            o_orderpriority="string",
            o_clerk="string",
            o_shippriority="int32",
            o_comment="string",
        ),
        name="tpch_orders",
    )


@pytest.fixture
def tpch(region, nation, customer, orders):
    fields_of_interest = [
        customer,
        region.r_name.name("region"),
        orders.o_totalprice,
        orders.o_orderdate.cast("timestamp").name("odate"),
    ]

    return (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)[fields_of_interest]
    )


def test_join_key_name(tpch, snapshot):
    year = tpch.odate.year().name("year")

    pre_sizes = tpch.group_by(year).size()
    t2 = tpch.view()
    conditional_avg = t2[t2.region == tpch.region].o_totalprice.mean().name("mean")
    amount_filter = tpch.o_totalprice > conditional_avg
    post_sizes = tpch[amount_filter].group_by(year).size()

    percent = (post_sizes[1] / pre_sizes[1].cast("double")).name("fraction")

    expr = pre_sizes.join(post_sizes, pre_sizes.year == post_sizes.year)[
        pre_sizes.year,
        pre_sizes[1].name("pre_count"),
        post_sizes[1].name("post_count"),
        percent,
    ]
    result = ibis.impala.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_join_key_name2(tpch, snapshot):
    year = tpch.odate.year().name("year")

    pre_sizes = tpch.group_by(year).size()
    post_sizes = tpch.group_by(year).size().view()

    expr = pre_sizes.join(post_sizes, pre_sizes.year == post_sizes.year)[
        pre_sizes.year,
        pre_sizes[1].name("pre_count"),
        post_sizes[1].name("post_count"),
    ]
    result = ibis.impala.compile(expr)
    snapshot.assert_match(result, "out.sql")


def test_group_by_with_window_preserves_range(snapshot):
    t = ibis.table(dict(one="string", two="double", three="int32"), name="my_data")
    w = ibis.cumulative_window(order_by=t.one)
    expr = t.group_by(t.three).mutate(four=t.two.sum().over(w))

    result = ibis.impala.compile(expr)
    snapshot.assert_match(result, "out.sql")
