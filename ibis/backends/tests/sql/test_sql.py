# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import operator
from operator import methodcaller

import pytest
from pytest import param

import ibis
from ibis.backends.tests.sql.conftest import to_sql
from ibis.tests.util import assert_decompile_roundtrip

pytestmark = pytest.mark.duckdb


@pytest.fixture(scope="module")
def star1():
    return ibis.table(
        [
            ("c", "int32"),
            ("f", "double"),
            ("foo_id", "string"),
            ("bar_id", "string"),
        ],
        name="star1",
    )


@pytest.fixture(scope="module")
def functional_alltypes():
    return ibis.table(
        {
            "id": "int32",
            "bool_col": "boolean",
            "tinyint_col": "int8",
            "smallint_col": "int16",
            "int_col": "int32",
            "bigint_col": "int64",
            "float_col": "float32",
            "double_col": "float64",
            "date_string_col": "string",
            "string_col": "string",
            "timestamp_col": "timestamp",
            "year": "int32",
            "month": "int32",
        },
        name="functional_alltypes",
    )


@pytest.fixture(scope="module")
def alltypes():
    return ibis.table(
        [
            ("a", "int8"),
            ("b", "int16"),
            ("c", "int32"),
            ("d", "int64"),
            ("e", "float32"),
            ("f", "float64"),
            ("g", "string"),
            ("h", "boolean"),
            ("i", "timestamp"),
            ("j", "date"),
            ("k", "time"),
        ],
        name="alltypes",
    )


@pytest.mark.parametrize("opname", ["ge", "gt", "lt", "le", "eq", "ne"])
def test_comparisons(functional_alltypes, opname, snapshot):
    op = getattr(operator, opname)
    expr = op(functional_alltypes.double_col, 5).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda d: (d > 0) & (d < 5), id="and"),
        param(lambda d: (d < 0) | (d > 5), id="or"),
    ],
)
def test_boolean_conjunction(functional_alltypes, expr_fn, snapshot):
    expr = expr_fn(functional_alltypes.double_col).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_between(functional_alltypes, snapshot):
    expr = functional_alltypes.double_col.between(5, 10).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("method_name", ["isnull", "notnull"])
def test_isnull_notnull(functional_alltypes, method_name, snapshot):
    expr = getattr(functional_alltypes.double_col, method_name)().name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_negate(functional_alltypes, snapshot):
    expr = -(functional_alltypes.double_col > 0)
    snapshot.assert_match(to_sql(expr.name("tmp")), "out.sql")


def test_coalesce(functional_alltypes, snapshot):
    d = functional_alltypes.double_col
    f = functional_alltypes.float_col

    expr = ibis.coalesce((d > 30).ifelse(d, ibis.NA), ibis.NA, f).name("tmp")
    snapshot.assert_match(to_sql(expr.name("tmp")), "out.sql")


def test_named_expr(functional_alltypes, snapshot):
    expr = functional_alltypes[(functional_alltypes.double_col * 2).name("foo")]
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        (lambda r, n: r.inner_join(n, r.r_regionkey == n.n_regionkey)),
        (lambda r, n: r.left_join(n, r.r_regionkey == n.n_regionkey)),
        (lambda r, n: r.outer_join(n, r.r_regionkey == n.n_regionkey)),
        (lambda r, n: r.inner_join(n, r.r_regionkey == n.n_regionkey).select(n)),
        (lambda r, n: r.left_join(n, r.r_regionkey == n.n_regionkey).select(n)),
        (lambda r, n: r.outer_join(n, r.r_regionkey == n.n_regionkey).select(n)),
    ],
    ids=["inner", "left", "outer", "inner_select", "left_select", "outer_select"],
)
def test_joins(region, nation, expr_fn, snapshot):
    expr = expr_fn(region, nation)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_join_just_materialized(nation, region, customer, snapshot):
    t1 = nation
    t2 = region
    t3 = customer
    joined = t1.inner_join(t2, t1.n_regionkey == t2.r_regionkey).inner_join(
        t3, t1.n_nationkey == t3.c_nationkey
    )  # GH #491

    snapshot.assert_match(to_sql(joined), "out.sql")


def test_full_outer_join(region, nation):
    """Testing full outer join separately due to previous issue with outer join
    resulting in left outer join (issue #1773)"""
    predicate = region.r_regionkey == nation.n_regionkey
    joined = region.outer_join(nation, predicate)
    joined_sql_str = to_sql(joined)
    assert "full" in joined_sql_str.lower()
    assert "left" not in joined_sql_str.lower()


def test_simple_case(simple_case, snapshot):
    expr = simple_case.name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_searched_case(search_case, snapshot):
    expr = search_case.name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_where_simple_comparisons(star1, snapshot):
    t1 = star1
    expr = t1.filter([t1.f > 0, t1.c < t1.f * 2])
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot)


@pytest.mark.parametrize(
    "expr_fn",
    [
        lambda t: t.agg([t["f"].sum().name("total")], [t["foo_id"]]),
        lambda t: t.agg([t["f"].sum().name("total")], ["foo_id", "bar_id"]),
        lambda t: t.agg(
            [t.f.sum().name("total")], by=["foo_id"], having=[t.f.sum() > 10]
        ),
        lambda t: t.agg(
            [t.f.sum().name("total")], by=["foo_id"], having=[t.count() > 100]
        ),
    ],
    ids=["single", "two", "having_sum", "having_count"],
)
def test_aggregate(star1, expr_fn, snapshot):
    expr = expr_fn(star1)
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    "key",
    ["f", ibis.random()],
    ids=["column", "random"],
)
def test_order_by(star1, key, snapshot):
    expr = star1.order_by(key)
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    "expr_fn", [methodcaller("limit", 10), methodcaller("limit", 10, offset=5)]
)
def test_limit(star1, expr_fn, snapshot):
    expr = expr_fn(star1)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_limit_filter(star1, snapshot):
    expr = star1[star1.f > 0].limit(10)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_limit_subquery(star1, snapshot):
    expr = star1.limit(10)[lambda x: x.f > 0]
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_cte_factor_distinct_but_equal(alltypes, snapshot):
    t = alltypes
    tt = alltypes.view()

    expr1 = t.group_by("g").aggregate(t.f.sum().name("metric"))
    expr2 = tt.group_by("g").aggregate(tt.f.sum().name("metric")).view()

    expr = expr1.join(expr2, expr1.g == expr2.g)[[expr1]]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_self_reference_join(star1, snapshot):
    t1 = star1
    t2 = t1.view()
    expr = t1.inner_join(t2, [t1.foo_id == t2.bar_id])[[t1]]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_self_reference_in_not_exists(functional_alltypes, snapshot):
    t = functional_alltypes
    t2 = t.view()

    cond = (t.string_col == t2.string_col).any()

    semi = t[cond]
    anti = t[-cond]

    snapshot.assert_match(to_sql(semi), "semi.sql")
    snapshot.assert_match(to_sql(anti), "anti.sql")


def test_where_uncorrelated_subquery(foo, bar, snapshot):
    expr = foo[foo.job.isin(bar.job)]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_where_correlated_subquery(foo, snapshot):
    t1 = foo
    t2 = t1.view()

    stat = t2[t1.dept_id == t2.dept_id].y.mean()
    expr = t1[t1.y > stat]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_subquery_aliased(star1, star2, snapshot):
    t1 = star1
    t2 = star2

    agged = t1.aggregate([t1.f.sum().name("total")], by=["foo_id"])
    expr = agged.inner_join(t2, [agged.foo_id == t2.foo_id])[agged, t2.value1]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_lower_projection_sort_key(star1, star2, snapshot):
    t1 = star1
    t2 = star2

    agged = t1.aggregate([t1.f.sum().name("total")], by=["foo_id"])
    expr = agged.inner_join(t2, [agged.foo_id == t2.foo_id])[agged, t2.value1]

    expr2 = expr[expr.total > 100].order_by(ibis.desc("total"))
    snapshot.assert_match(to_sql(expr2), "out.sql")
    assert_decompile_roundtrip(expr2, snapshot)


def test_exists(foo_t, bar_t, snapshot):
    t1 = foo_t
    t2 = bar_t
    cond = (t1.key1 == t2.key1).any()
    e1 = t1[cond]

    snapshot.assert_match(to_sql(e1), "e1.sql")

    cond2 = ((t1.key1 == t2.key1) & (t2.key2 == "foo")).any()
    e2 = t1[cond2]

    snapshot.assert_match(to_sql(e2), "e2.sql")


def test_not_exists(not_exists, snapshot):
    snapshot.assert_match(to_sql(not_exists), "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda t: t.distinct(), id="table_distinct"),
        param(
            lambda t: t["string_col", "int_col"].distinct(), id="projection_distinct"
        ),
        param(
            lambda t: t[t.string_col].distinct(), id="single_column_projection_distinct"
        ),
        param(lambda t: t.int_col.nunique().name("nunique"), id="count_distinct"),
        param(
            lambda t: t.group_by("string_col").aggregate(
                t.int_col.nunique().name("nunique")
            ),
            id="group_by_count_distinct",
        ),
    ],
)
def test_distinct(functional_alltypes, expr_fn, snapshot):
    expr = expr_fn(functional_alltypes)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_sort_aggregation_translation_failure(functional_alltypes, snapshot):
    # This works around a nuance with our choice to hackishly fuse SortBy
    # after Aggregate to produce a single select statement rather than an
    # inline view.
    t = functional_alltypes

    agg = t.group_by("string_col").aggregate(t.double_col.max().name("foo"))
    expr = agg.order_by(ibis.desc("foo"))

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_where_correlated_subquery_with_join(snapshot):
    # GH3163
    # ibis code
    part = ibis.table([("p_partkey", "int64")], name="part")
    partsupp = ibis.table(
        [
            ("ps_partkey", "int64"),
            ("ps_supplycost", "float64"),
            ("ps_suppkey", "int64"),
        ],
        name="partsupp",
    )
    supplier = ibis.table([("s_suppkey", "int64")], name="supplier")

    q = part.join(partsupp, part.p_partkey == partsupp.ps_partkey)
    q = q[
        part.p_partkey,
        partsupp.ps_supplycost,
    ]
    subq = partsupp.join(supplier, supplier.s_suppkey == partsupp.ps_suppkey)
    subq = subq.select(partsupp.ps_partkey, partsupp.ps_supplycost)
    subq = subq[subq.ps_partkey == q.p_partkey]

    expr = q[q.ps_supplycost == subq.ps_supplycost.min()]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_mutate_filter_join_no_cross_join(snapshot):
    person = ibis.table(
        [("person_id", "int64"), ("birth_datetime", "timestamp")],
        name="person",
    )
    mutated = person.mutate(age=400)
    expr = mutated.filter(mutated.age <= 40)[mutated.person_id]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_filter_group_by_agg_with_same_name(snapshot):
    # GH 2907
    t = ibis.table([("int_col", "int32"), ("bigint_col", "int64")], name="t")
    expr = (
        t.group_by("int_col")
        .aggregate(bigint_col=lambda t: t.bigint_col.sum())
        .filter(lambda t: t.bigint_col == 60)
    )
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.fixture
def person():
    return ibis.table(
        dict(id="string", personal="string", family="string"), name="person"
    )


@pytest.fixture
def visited():
    return ibis.table(dict(id="int32", site="string", dated="string"), name="visited")


@pytest.fixture
def survey():
    return ibis.table(
        dict(taken="int32", person="string", quant="string", reading="float32"),
        name="survey",
    )


def test_no_cross_join(person, visited, survey, snapshot):
    expr = person.join(survey, person.id == survey.person).join(
        visited, visited.id == survey.taken
    )
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.fixture
def test1():
    return ibis.table(name="test1", schema=dict(id1="int32", val1="float32"))


@pytest.fixture
def test2():
    return ibis.table(
        name="test2",
        schema=ibis.schema(dict(id2a="int32", id2b="int64", val2="float64")),
    )


@pytest.fixture
def test3():
    return ibis.table(
        name="test3",
        schema=dict(id3="string", val2="float64", dt="timestamp"),
    )


def test_gh_1045(test1, test2, test3, snapshot):
    t1 = test1
    t2 = test2
    t3 = test3

    t3 = t3.mutate(id3=t3.id3.cast("int64"))

    t3 = t3.mutate(t3_val2=t3.id3)
    t4 = t3.join(t2, t2.id2b == t3.id3)

    t1 = t1[[t1[c].name(f"t1_{c}") for c in t1.columns]]

    expr = t1.left_join(t4, t1.t1_id1 == t4.id2a)

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_multi_join(snapshot):
    t1 = ibis.table(dict(x1="int64", y1="int64"), name="t1")
    t2 = ibis.table(dict(x2="int64"), name="t2")
    t3 = ibis.table(dict(x3="int64", y2="int64"), name="t3")
    t4 = ibis.table(dict(x4="int64"), name="t4")

    j1 = t1.join(t2, t1.x1 == t2.x2)
    j2 = t3.join(t4, t3.x3 == t4.x4)
    expr = j1.join(j2, j1.y1 == j2.y2)

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_no_cart_join(snapshot):
    facts = ibis.table(dict(product_id="!int32"), name="facts")
    products = ibis.table(
        dict(
            ancestor_level_name="string",
            ancestor_level_number="int32",
            ancestor_node_sort_order="int64",
            descendant_node_natural_key="int32",
        ),
        name="products",
    )

    products = products.mutate(
        product_level_name=lambda t: ibis.literal("-")
        .lpad(((t.ancestor_level_number - 1) * 7), "-")
        .concat(t.ancestor_level_name)
    )

    predicate = facts.product_id == products.descendant_node_natural_key
    joined = facts.join(products, predicate)

    gb = joined.group_by(products.ancestor_node_sort_order)
    agg = gb.aggregate(n=ibis.literal(1))
    ob = agg.order_by(products.ancestor_node_sort_order)

    snapshot.assert_match(to_sql(ob), "out.sql")


def test_order_by_expr(snapshot):
    t = ibis.table(dict(a="int", b="string"), name="t")
    expr = t[lambda t: t.a == 1].order_by(lambda t: t.b + "a")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_no_cartesian_join(snapshot):
    customers = ibis.table(
        dict(customer_id="int64", first_name="string", last_name="string"),
        name="customers",
    )
    orders = ibis.table(
        dict(order_id="int64", customer_id="int64", order_date="date", status="string"),
        name="orders",
    )
    payments = ibis.table(
        dict(
            payment_id="int64",
            order_id="int64",
            payment_method="string",
            amount="float64",
        ),
        name="payments",
    )

    customer_orders = orders.group_by("customer_id").aggregate(
        first_order=orders.order_date.min(),
        most_recent_order=orders.order_date.max(),
        number_of_orders=orders.order_id.count(),
    )

    customer_payments = (
        payments.left_join(orders, "order_id")
        .group_by(orders.customer_id)
        .aggregate(total_amount=payments.amount.sum())
    )

    final = (
        customers.left_join(customer_orders, "customer_id")
        .drop("customer_id_right")
        .left_join(customer_payments, "customer_id")[
            customers.customer_id,
            customers.first_name,
            customers.last_name,
            customer_orders.first_order,
            customer_orders.most_recent_order,
            customer_orders.number_of_orders,
            customer_payments.total_amount.name("customer_lifetime_value"),
        ]
    )
    snapshot.assert_match(ibis.to_sql(final, dialect="duckdb"), "out.sql")
