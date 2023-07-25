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
import sqlglot as sg
from pytest import param
from sqlalchemy import types as sat

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import AlchemyCompiler, BaseAlchemyBackend
from ibis.backends.base.sql.alchemy.datatypes import AlchemyType, ArrayType
from ibis.tests.expr.mocks import MockAlchemyBackend
from ibis.tests.util import assert_decompile_roundtrip, assert_equal

sa = pytest.importorskip("sqlalchemy")


L = sa.literal


def to_sql(expr, *args, **kwargs) -> str:
    compiled = AlchemyCompiler.to_sql(expr, *args, **kwargs)
    sqlstring = str(compiled.compile(compile_kwargs=dict(literal_binds=True)))
    return sg.parse_one(sqlstring).sql(pretty=True, dialect="duckdb")


@pytest.fixture(scope="module")
def con():
    return MockAlchemyBackend()


@pytest.fixture(scope="module")
def star1(con):
    return con.table("star1")


@pytest.fixture(scope="module")
def functional_alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="module")
def alltypes(con):
    return con.table("alltypes")


def test_sqla_schema_conversion():
    typespec = [
        # name, type, nullable
        ("smallint", sat.SMALLINT, False, dt.int16),
        ("smallint_", sat.SmallInteger, False, dt.int16),
        ("int", sat.INTEGER, True, dt.int32),
        ("integer", sat.INTEGER, True, dt.int32),
        ("integer_", sat.Integer, True, dt.int32),
        ("bigint", sat.BIGINT, False, dt.int64),
        ("bigint_", sat.BigInteger, False, dt.int64),
        ("real", sat.REAL, True, dt.float32),
        ("bool", sat.BOOLEAN, True, dt.bool),
        ("bool_", sat.Boolean, True, dt.bool),
        ("timestamp", sat.DATETIME, True, dt.timestamp),
        ("timestamp_", sat.DateTime, True, dt.timestamp),
    ]

    sqla_types = []
    ibis_types = []
    for name, t, nullable, ibis_type in typespec:
        sqla_types.append(sa.Column(name, t, nullable=nullable))
        ibis_types.append((name, ibis_type(nullable=nullable)))

    table = sa.Table("tname", sa.MetaData(), *sqla_types)

    schema = BaseAlchemyBackend._schema_from_sqla_table(table)
    expected = ibis.schema(ibis_types)

    assert_equal(schema, expected)


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
def test_joins(con, expr_fn, snapshot):
    region = con.table("tpch_region")
    nation = con.table("tpch_nation")

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


def test_full_outer_join(con):
    """Testing full outer join separately due to previous issue with outer join
    resulting in left outer join (issue #1773)"""
    region = con.table("tpch_region")
    nation = con.table("tpch_nation")

    predicate = region.r_regionkey == nation.n_regionkey
    joined = region.outer_join(nation, predicate)
    joined_sql_str = str(joined.compile())
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


def test_cte_factor_distinct_but_equal(con, snapshot):
    t = con.table("alltypes")
    tt = con.table("alltypes")

    expr1 = t.group_by("g").aggregate(t.f.sum().name("metric"))
    expr2 = tt.group_by("g").aggregate(tt.f.sum().name("metric")).view()

    expr = expr1.join(expr2, expr1.g == expr2.g)[[expr1]]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_self_reference_join(star1, snapshot):
    t1 = star1
    t2 = t1.view()
    expr = t1.inner_join(t2, [t1.foo_id == t2.bar_id])[[t1]]

    snapshot.assert_match(to_sql(expr), "out.sql")


def test_self_reference_in_not_exists(con, snapshot):
    t = con.table("functional_alltypes")
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


def test_exists(con, foo_t, bar_t, snapshot):
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

    t3 = t3[[c for c in t3.columns if c != "id3"]].mutate(id3=t3.id3.cast("int64"))

    t3 = t3[[c for c in t3.columns if c != "val2"]].mutate(t3_val2=t3.id3)
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


def test_tpc_h11(snapshot):
    NATION = "GERMANY"
    FRACTION = 0.0001

    partsupp = ibis.table(
        dict(
            ps_partkey="int32",
            ps_suppkey="int32",
            ps_availqty="int32",
            ps_supplycost="decimal(15, 2)",
        ),
        name="partsupp",
    )
    supplier = ibis.table(
        dict(s_suppkey="int32", s_nationkey="int32"),
        name="supplier",
    )
    nation = ibis.table(
        dict(n_nationkey="int32", n_name="string"),
        name="nation",
    )

    q = partsupp
    q = q.join(supplier, partsupp.ps_suppkey == supplier.s_suppkey)
    q = q.join(nation, nation.n_nationkey == supplier.s_nationkey)

    q = q.filter([q.n_name == NATION])

    innerq = partsupp
    innerq = innerq.join(supplier, partsupp.ps_suppkey == supplier.s_suppkey)
    innerq = innerq.join(nation, nation.n_nationkey == supplier.s_nationkey)
    innerq = innerq.filter([innerq.n_name == NATION])
    innerq = innerq.aggregate(total=(innerq.ps_supplycost * innerq.ps_availqty).sum())

    gq = q.group_by([q.ps_partkey])
    q = gq.aggregate(value=(q.ps_supplycost * q.ps_availqty).sum())
    q = q.filter([q.value > innerq.total * FRACTION])
    q = q.order_by(ibis.desc(q.value))

    snapshot.assert_match(to_sql(q), "out.sql")


def test_to_sqla_type_array_of_non_primitive():
    result = AlchemyType.from_ibis(dt.Array(dt.Struct(dict(a="int"))))
    [(result_name, result_type)] = result.value_type.fields.items()
    expected_name = "a"
    assert result_name == expected_name
    assert type(result_type) == sat.BigInteger
    assert isinstance(result, ArrayType)


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


def test_tpc_h17(snapshot):
    BRAND = "Brand#23"
    CONTAINER = "MED BOX"

    lineitem = ibis.table(
        dict(
            l_partkey="!int32", l_quantity="!int32", l_extendedprice="!decimal(15, 2)"
        ),
        name="lineitem",
    )
    part = ibis.table(
        dict(p_partkey="!int32", p_brand="!string", p_container="!string"), name="part"
    )

    q = lineitem.join(part, part.p_partkey == lineitem.l_partkey)
    innerq = lineitem.filter([lineitem.l_partkey == q.p_partkey])
    q = q.filter(
        [
            q.p_brand == BRAND,
            q.p_container == CONTAINER,
            q.l_quantity < (0.2 * innerq.l_quantity.mean()),
        ]
    )
    q = q.aggregate(
        avg_yearly=q.l_extendedprice.sum() / ibis.literal(7.0, type="decimal(15, 2)")
    )

    snapshot.assert_match(to_sql(q), "out.sql")
