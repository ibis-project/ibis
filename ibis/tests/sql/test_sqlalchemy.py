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

import operator

import pytest
from pytest import param
from sqlalchemy import func as F
from sqlalchemy import sql
from sqlalchemy import types as sat
from sqlalchemy.engine.default import DefaultDialect

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyContext,
    schema_from_table,
    to_sqla_type,
)
from ibis.backends.base.sql.alchemy.datatypes import ArrayType
from ibis.tests.expr.mocks import MockAlchemyBackend
from ibis.tests.util import assert_decompile_roundtrip, assert_equal

sa = pytest.importorskip('sqlalchemy')


L = sa.literal


def to_sql(expr, *args, **kwargs) -> str:
    compiled = AlchemyCompiler.to_sql(expr, *args, **kwargs)
    return str(compiled.compile(compile_kwargs=dict(literal_binds=True)))


@pytest.fixture(scope="module")
def con():
    return MockAlchemyBackend()


@pytest.fixture(scope="module")
def star1(con):
    return con.table("star1")


@pytest.fixture(scope="module")
def sa_star1(con, star1):
    return con.meta.tables["star1"]


@pytest.fixture(scope="module")
def functional_alltypes(con):
    return con.table('functional_alltypes')


@pytest.fixture(scope="module")
def sa_functional_alltypes(con, functional_alltypes):
    return con.meta.tables['functional_alltypes'].alias("t0")


@pytest.fixture(scope="module")
def alltypes(con):
    return con.table('alltypes')


@pytest.fixture(scope="module")
def sa_alltypes(con, alltypes):
    return con.meta.tables['alltypes'].alias("t0")


def _check(expr, sqla):
    context = AlchemyContext(compiler=AlchemyCompiler)
    result_sqla = AlchemyCompiler.to_sql(expr, context)
    result = str(result_sqla.compile(compile_kwargs=dict(literal_binds=True)))
    expected = str(sqla.compile(compile_kwargs=dict(literal_binds=True)))
    assert result == expected


def test_sqla_schema_conversion(con):
    typespec = [
        # name, type, nullable
        ('smallint', sat.SmallInteger, False, dt.int16),
        ('int', sat.Integer, True, dt.int32),
        ('integer', sat.INTEGER(), True, dt.int32),
        ('bigint', sat.BigInteger, False, dt.int64),
        ('real', sat.REAL, True, dt.float32),
        ('bool', sat.Boolean, True, dt.bool),
        ('timestamp', sat.DateTime, True, dt.timestamp),
    ]

    sqla_types = []
    ibis_types = []
    for name, t, nullable, ibis_type in typespec:
        sqla_types.append(sa.Column(name, t, nullable=nullable))
        ibis_types.append((name, ibis_type(nullable=nullable)))

    table = sa.Table('tname', con.meta, *sqla_types)

    schema = schema_from_table(table)
    expected = ibis.schema(ibis_types)

    assert_equal(schema, expected)


@pytest.mark.parametrize(
    "op",
    [
        operator.ge,
        operator.gt,
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
    ],
)
def test_comparisons(con, functional_alltypes, sa_functional_alltypes, op):
    expr = op(functional_alltypes.double_col, 5).name("tmp")
    expected = sa.select(op(sa_functional_alltypes.c.double_col, L(5)).label("tmp"))
    _check(expr, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        (
            lambda d: (d > 0) & (d < 5),
            lambda sd: sql.and_(sd > L(0), sd < L(5)),
        ),
        (
            lambda d: (d < 0) | (d > 5),
            lambda sd: sql.or_(sd < L(0), sd > L(5)),
        ),
    ],
)
def test_boolean_conjunction(
    con,
    sa_functional_alltypes,
    functional_alltypes,
    expr_fn,
    expected_fn,
):
    expr = expr_fn(functional_alltypes.double_col).name('tmp')
    expected = sa.select(expr_fn(sa_functional_alltypes.c.double_col).label("tmp"))
    _check(expr, expected)


def test_between(con, functional_alltypes, sa_functional_alltypes):
    expr = functional_alltypes.double_col.between(5, 10).name("tmp")
    expected = sa.select(
        sa_functional_alltypes.c.double_col.between(L(5), L(10)).label("tmp")
    )
    _check(expr, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(lambda d: d.isnull(), lambda sd: sd.is_(sa.null()), id="isnull"),
        param(lambda d: d.notnull(), lambda sd: sd.is_not(sa.null()), id="notnull"),
    ],
)
def test_isnull_notnull(
    con,
    sa_functional_alltypes,
    functional_alltypes,
    expr_fn,
    expected_fn,
):
    expr = expr_fn(functional_alltypes.double_col).name("tmp")
    expected = sa.select(expected_fn(sa_functional_alltypes.c.double_col).label("tmp"))
    _check(expr, expected)


def test_negate(sa_functional_alltypes, functional_alltypes):
    expr = -(functional_alltypes.double_col > 0)
    expected = sa.select(
        sql.not_(sa_functional_alltypes.c.double_col > L(0)).label("tmp")
    )
    _check(expr.name('tmp'), expected)


def test_coalesce(sa_functional_alltypes, functional_alltypes):
    sat = sa_functional_alltypes
    sd = sat.c.double_col
    sf = sat.c.float_col

    d = functional_alltypes.double_col
    f = functional_alltypes.float_col
    null = sa.null()

    v1 = ibis.NA
    v2 = (d > 30).ifelse(d, ibis.NA)
    v3 = f

    expr = ibis.coalesce(v2, v1, v3).name("tmp")
    expected = sa.select(
        sa.func.coalesce(
            sa.case((sd > L(30), sd), else_=null),
            null,
            sf,
        ).label("tmp")
    )
    _check(expr, expected)


def test_named_expr(sa_functional_alltypes, functional_alltypes):
    expr = functional_alltypes[(functional_alltypes.double_col * 2).name('foo')]
    expected = sa.select((sa_functional_alltypes.c.double_col * L(2)).label('foo'))
    _check(expr, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        (
            lambda r, n: r.inner_join(n, r.r_regionkey == n.n_regionkey),
            lambda rt, nt: rt.join(
                nt,
                rt.c.r_regionkey == nt.c.n_regionkey,
            ).select(),
        ),
        (
            lambda r, n: r.left_join(n, r.r_regionkey == n.n_regionkey),
            lambda rt, nt: rt.join(
                nt,
                rt.c.r_regionkey == nt.c.n_regionkey,
                isouter=True,
            ).select(),
        ),
        (
            lambda r, n: r.outer_join(n, r.r_regionkey == n.n_regionkey),
            lambda rt, nt: rt.outerjoin(
                nt,
                rt.c.r_regionkey == nt.c.n_regionkey,
                full=True,
            ).select(),
        ),
        (
            lambda r, n: r.inner_join(n, r.r_regionkey == n.n_regionkey).projection(n),
            lambda rt, nt: sa.select(nt).select_from(
                rt.join(nt, rt.c.r_regionkey == nt.c.n_regionkey)
            ),
        ),
        (
            lambda r, n: r.left_join(n, r.r_regionkey == n.n_regionkey).projection(n),
            lambda rt, nt: sa.select(nt).select_from(
                rt.join(
                    nt,
                    rt.c.r_regionkey == nt.c.n_regionkey,
                    isouter=True,
                )
            ),
        ),
        (
            lambda r, n: r.outer_join(n, r.r_regionkey == n.n_regionkey).projection(n),
            lambda rt, nt: sa.select(nt).select_from(
                rt.outerjoin(
                    nt,
                    rt.c.r_regionkey == nt.c.n_regionkey,
                    full=True,
                )
            ),
        ),
    ],
)
def test_joins(con, expr_fn, expected_fn):
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')

    rt = con.meta.tables["tpch_region"].alias("t0")
    nt = con.meta.tables["tpch_nation"].alias("t1")

    expr = expr_fn(region, nation)
    expected = expected_fn(rt, nt)
    _check(expr, expected)


def test_join_just_materialized(con, nation, region, customer):
    t1 = nation
    t2 = region
    t3 = customer
    joined = t1.inner_join(t2, t1.n_regionkey == t2.r_regionkey).inner_join(
        t3, t1.n_nationkey == t3.c_nationkey
    )  # GH #491

    nt, rt, ct = (
        con.meta.tables[name].alias(f"t{i:d}")
        for i, name in enumerate(['tpch_nation', 'tpch_region', 'tpch_customer'])
    )

    sqla_joined = nt.join(rt, nt.c.n_regionkey == rt.c.r_regionkey).join(
        ct, nt.c.n_nationkey == ct.c.c_nationkey
    )

    expected = sa.select(sqla_joined)

    _check(joined, expected)


def test_full_outer_join(con):
    """Testing full outer join separately due to previous issue with outer join
    resulting in left outer join (issue #1773)"""
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')

    predicate = region.r_regionkey == nation.n_regionkey
    joined = region.outer_join(nation, predicate)
    joined_sql_str = str(joined.compile())
    assert 'full' in joined_sql_str.lower()
    assert 'left' not in joined_sql_str.lower()


def test_simple_case(sa_alltypes, simple_case):
    st = sa_alltypes
    expr = simple_case.name("tmp")
    expected = sa.select(
        sa.case(
            (st.c.g == L('foo'), L('bar')),
            (st.c.g == L('baz'), L('qux')),
            else_='default',
        ).label("tmp")
    )
    _check(expr, expected)


def test_searched_case(sa_alltypes, search_case):
    st = sa_alltypes.alias("t0")
    expr = search_case.name("tmp")
    expected = sa.select(
        sa.case(
            (st.c.f > L(0), st.c.d * L(2)),
            (st.c.c < L(0), st.c.a * L(2)),
            else_=sa.cast(sa.null(), sa.BIGINT),
        ).label("tmp")
    )
    _check(expr, expected)


def test_where_simple_comparisons(sa_star1, star1, snapshot):
    t1 = star1
    expr = t1.filter([t1.f > 0, t1.c < t1.f * 2])
    st = sa_star1.alias("t0")
    expected = sa.select(st).where(sql.and_(st.c.f > L(0), st.c.c < (st.c.f * L(2))))
    _check(expr, expected)
    snapshot.assert_match(to_sql(expr), "out.sql")
    assert_decompile_roundtrip(expr, snapshot)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        (
            lambda t: t.aggregate([t['f'].sum().name('total')], [t['foo_id']]),
            lambda st: sa.select(st.c.foo_id, F.sum(st.c.f).label('total')).group_by(
                st.c.foo_id
            ),
        ),
        (
            lambda t: t.aggregate([t['f'].sum().name('total')], ['foo_id', 'bar_id']),
            lambda st: sa.select(
                st.c.foo_id, st.c.bar_id, F.sum(st.c.f).label('total')
            ).group_by(st.c.foo_id, st.c.bar_id),
        ),
        (
            lambda t: t.aggregate(
                [t.f.sum().name("total")],
                by=["foo_id"],
                having=[t.f.sum() > 10],
            ),
            lambda st: (
                sa.select(st.c.foo_id, F.sum(st.c.f).label("total"))
                .group_by(st.c.foo_id)
                .having(F.sum(st.c.f).label("total") > L(10))
            ),
        ),
        (
            lambda t: t.aggregate(
                [t.f.sum().name("total")],
                by=["foo_id"],
                having=[t.count() > 100],
            ),
            lambda st: (
                sa.select(st.c.foo_id, F.sum(st.c.f).label("total"))
                .group_by(st.c.foo_id)
                .having(F.count() > L(100))
            ),
        ),
    ],
)
def test_aggregate(con, star1, sa_star1, expr_fn, expected_fn):
    st = sa_star1.alias('t0')
    expr = expr_fn(star1)
    expected = expected_fn(st)
    _check(expr, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        pytest.param(
            lambda t: t.order_by("f"),
            lambda b: sa.select(b).order_by(b.c.f.asc()),
            id="order_by",
        ),
        pytest.param(
            lambda t: t.order_by(("f", 0)),
            lambda b: sa.select(b).order_by(b.c.f.desc()),
            id="order_by_ascending",
        ),
        pytest.param(
            lambda t: t.order_by(["c", ("f", 0)]),
            lambda b: sa.select(b).order_by(b.c.c.asc(), b.c.f.desc()),
            id="order_by_mixed",
        ),
        pytest.param(
            lambda t: t.order_by(ibis.random()),
            lambda b: sa.select(b).order_by(sa.func.random().asc()),
            id="order_by_random",
        ),
    ],
)
def test_order_by(con, star1, sa_star1, expr_fn, expected_fn):
    st = sa_star1.alias("t0")
    expr = expr_fn(star1)
    expected = expected_fn(st)
    _check(expr, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        (lambda star1: star1.limit(10), lambda st: st.limit(10)),
        (
            lambda star1: star1.limit(10, offset=5),
            lambda st: st.limit(10).offset(5),
        ),
    ],
)
def test_limit(star1, sa_star1, expr_fn, expected_fn):
    expr = expr_fn(star1)
    expected = expected_fn(sa.select(sa_star1.alias("t0")))
    _check(expr, expected)


def test_limit_filter(con, star1, sa_star1):
    expr = star1[star1.f > 0].limit(10)
    expected = sa_star1.alias("t0")
    expected = sa.select(expected).where(expected.c.f > L(0)).limit(10)
    _check(expr, expected)


def test_limit_subquery(con, star1, sa_star1):
    expr = star1.limit(10)[lambda x: x.f > 0]
    expected = sa.select(sa_star1.alias("t1")).limit(10).alias("t0")
    expected = sa.select(expected).where(expected.c.f > 0)
    _check(expr, expected)


def test_cte_factor_distinct_but_equal(con, sa_alltypes, snapshot):
    t = con.table('alltypes')
    tt = con.table('alltypes')

    expr1 = t.group_by('g').aggregate(t.f.sum().name('metric'))
    expr2 = tt.group_by('g').aggregate(tt.f.sum().name('metric')).view()

    expr = expr1.join(expr2, expr1.g == expr2.g)[[expr1]]

    #

    t2 = sa_alltypes.alias('t2')
    t0 = sa.select(t2.c.g, F.sum(t2.c.f).label('metric')).group_by(t2.c.g).cte('t0')

    t1 = t0.alias('t1')
    table_set = t0.join(t1, t0.c.g == t1.c.g)
    stmt = sa.select(t0).select_from(table_set)

    _check(expr, stmt)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_self_reference_join(star1, sa_star1, snapshot):
    t1 = star1
    t2 = t1.view()
    expr = t1.inner_join(t2, [t1.foo_id == t2.bar_id])[[t1]]
    #
    t0 = sa_star1.alias('t0')
    t1 = sa_star1.alias('t1')

    table_set = t0.join(t1, t0.c.foo_id == t1.c.bar_id)
    expected = sa.select(t0).select_from(table_set)
    _check(expr, expected)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_self_reference_in_not_exists(con, sa_functional_alltypes, snapshot):
    t = con.table('functional_alltypes')
    t2 = t.view()

    cond = (t.string_col == t2.string_col).any()
    semi = t[cond]
    anti = t[-cond]

    s1 = sa_functional_alltypes.alias('t0')
    s2 = sa_functional_alltypes.alias('t1')

    cond = sa.exists(L(1)).select_from(s1).where(s1.c.string_col == s2.c.string_col)

    ex_semi = sa.select(s1).where(cond)
    ex_anti = sa.select(s1).where(~cond)

    _check(semi, ex_semi)
    snapshot.assert_match(to_sql(semi), "semi.sql")

    _check(anti, ex_anti)
    snapshot.assert_match(to_sql(anti), "anti.sql")


def test_where_uncorrelated_subquery(con, foo, bar, snapshot):
    expr = foo[foo.job.isin(bar.job)]
    #
    foo = con.meta.tables["foo"].alias("t0")
    bar = con.meta.tables["bar"]

    subq = sa.select(bar.c.job)
    stmt = sa.select(foo).where(foo.c.job.in_(subq))
    _check(expr, stmt)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_where_correlated_subquery(con, foo, snapshot):
    t1 = foo
    t2 = t1.view()

    stat = t2[t1.dept_id == t2.dept_id].y.mean()
    expr = t1[t1.y > stat]
    #
    foo = con.meta.tables["foo"]
    t0 = foo.alias('t0')
    t1 = foo.alias('t1')
    subq = sa.select(F.avg(t1.c.y).label("Mean(y)")).where(t0.c.dept_id == t1.c.dept_id)
    # For versions of SQLAlchemy where scalar_subquery exists,
    # it should be used (otherwise, a deprecation warning is raised)
    if hasattr(subq, 'scalar_subquery'):
        subq = subq.scalar_subquery()
    stmt = sa.select(t0).where(t0.c.y > subq)
    _check(expr, stmt)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_subquery_aliased(con, star1, star2, snapshot):
    t1 = star1
    t2 = star2

    agged = t1.aggregate([t1.f.sum().name('total')], by=['foo_id'])
    expr = agged.inner_join(t2, [agged.foo_id == t2.foo_id])[agged, t2.value1]
    #
    s1 = con.meta.tables["star1"].alias("t2")
    s2 = con.meta.tables["star2"].alias("t1")

    agged = (
        sa.select(s1.c.foo_id, F.sum(s1.c.f).label('total'))
        .group_by(s1.c.foo_id)
        .alias('t0')
    )

    joined = agged.join(s2, agged.c.foo_id == s2.c.foo_id)
    expected = sa.select(agged, s2.c.value1).select_from(joined)

    _check(expr, expected)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_lower_projection_sort_key(con, star1, star2, snapshot):
    t1 = star1
    t2 = star2

    agged = t1.aggregate([t1.f.sum().name('total')], by=['foo_id'])
    expr = agged.inner_join(t2, [agged.foo_id == t2.foo_id])[agged, t2.value1]
    #
    t4 = con.meta.tables["star1"].alias("t4")
    t3 = con.meta.tables["star2"].alias("t3")

    t2 = (
        sa.select(t4.c.foo_id, F.sum(t4.c.f).label('total'))
        .group_by(t4.c.foo_id)
        .alias('t2')
    )
    t1 = (
        sa.select(t2.c.foo_id, t2.c.total, t3.c.value1)
        .select_from(t2.join(t3, t2.c.foo_id == t3.c.foo_id))
        .alias('t1')
    )
    t0 = (
        sa.select(t1.c.foo_id, t1.c.total, t1.c.value1)
        .where(t1.c.total > L(100))
        .alias('t0')
    )
    expected = sa.select(t0.c.foo_id, t0.c.total, t0.c.value1).order_by(
        t0.c.total.desc()
    )

    expr2 = expr[expr.total > 100].order_by(ibis.desc('total'))
    _check(expr2, expected)
    snapshot.assert_match(to_sql(expr2), "out.sql")
    assert_decompile_roundtrip(expr2, snapshot)


def test_exists(con, foo_t, bar_t, snapshot):
    t1 = foo_t
    t2 = bar_t
    cond = (t1.key1 == t2.key1).any()
    e1 = t1[cond]

    cond2 = ((t1.key1 == t2.key1) & (t2.key2 == 'foo')).any()
    e2 = t1[cond2]
    #
    t1 = con.meta.tables["foo_t"].alias("t0")
    t2 = con.meta.tables["bar_t"].alias("t1")

    cond1 = sa.exists(L(1)).where(t1.c.key1 == t2.c.key1)
    ex1 = sa.select(t1).where(cond1)

    cond2 = sa.exists(L(1)).where(
        sql.and_(t1.c.key1 == t2.c.key1, t2.c.key2 == L('foo'))
    )
    ex2 = sa.select(t1).where(cond2)

    _check(e1, ex1)
    snapshot.assert_match(to_sql(e1), "e1.sql")

    _check(e2, ex2)
    snapshot.assert_match(to_sql(e2), "e2.sql")


def test_not_exists(con, not_exists, snapshot):
    t1 = con.table("t1")
    t2 = con.table("t2")

    expr = not_exists

    t1 = con.meta.tables["foo_t"].alias("t0")
    t2 = con.meta.tables["bar_t"].alias("t1")

    expected = sa.select(t1).where(
        sa.not_(sa.exists(L(1)).where(t1.c.key1 == t2.c.key1))
    )

    _check(expr, expected)
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        (lambda t: t.distinct(), lambda sat: sa.select(sat).distinct()),
        (
            lambda t: t['string_col', 'int_col'].distinct(),
            lambda sat: sa.select(sat.c.string_col, sat.c.int_col).distinct(),
        ),
        (
            lambda t: t[t.string_col].distinct(),
            lambda sat: sa.select(sat.c.string_col.distinct()),
        ),
        (
            lambda t: t.int_col.nunique().name('nunique'),
            lambda sat: sa.select(F.count(sat.c.int_col.distinct()).label('nunique')),
        ),
        (
            lambda t: t.group_by('string_col').aggregate(
                t.int_col.nunique().name('nunique')
            ),
            lambda sat: sa.select(
                sat.c.string_col,
                F.count(sat.c.int_col.distinct()).label('nunique'),
            ).group_by(sat.c.string_col),
        ),
    ],
)
def test_distinct(
    sa_functional_alltypes,
    functional_alltypes,
    expr_fn,
    expected_fn,
):
    expr = expr_fn(functional_alltypes)
    expected = expected_fn(sa_functional_alltypes)
    _check(expr, expected)


def test_sort_aggregation_translation_failure(
    sa_functional_alltypes,
    functional_alltypes,
):
    # This works around a nuance with our choice to hackishly fuse SortBy
    # after Aggregate to produce a single select statement rather than an
    # inline view.
    t = functional_alltypes

    agg = t.group_by('string_col').aggregate(t.double_col.max().name('foo'))
    expr = agg.order_by(ibis.desc('foo'))

    sat = sa_functional_alltypes.alias("t1")
    base = (
        sa.select(sat.c.string_col, F.max(sat.c.double_col).label('foo')).group_by(
            sat.c.string_col
        )
    ).alias('t0')

    ex = (
        sa.select(base.c.string_col, base.c.foo)
        .select_from(base)
        .order_by(sa.desc('foo'))
    )

    _check(expr, ex)


def test_where_correlated_subquery_with_join():
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
    subq = subq.projection([partsupp.ps_partkey, partsupp.ps_supplycost])
    subq = subq[subq.ps_partkey == q.p_partkey]

    expr = q[q.ps_supplycost == subq.ps_supplycost.min()]

    # sqlalchemy code
    part = sa.table("part", sa.column("p_partkey"))
    supplier = sa.table("supplier", sa.column("s_suppkey"))
    partsupp = sa.table(
        "partsupp",
        sa.column("ps_partkey"),
        sa.column("ps_supplycost"),
        sa.column("ps_suppkey"),
    )

    part_t1 = part.alias("t1")
    partsupp_t2 = partsupp.alias("t2")

    t0 = (
        sa.select(part_t1.c.p_partkey, partsupp_t2.c.ps_supplycost)
        .select_from(
            part_t1.join(
                partsupp_t2,
                onclause=part_t1.c.p_partkey == partsupp_t2.c.ps_partkey,
            )
        )
        .alias("t0")
    )

    partsupp_t2 = partsupp.alias("t2")
    supplier_t3 = supplier.alias("t3")
    t1 = (
        sa.select(partsupp_t2.c.ps_partkey, partsupp_t2.c.ps_supplycost)
        .select_from(
            partsupp_t2.join(
                supplier_t3,
                onclause=supplier_t3.c.s_suppkey == partsupp_t2.c.ps_suppkey,
            )
        )
        .alias("t1")
    )
    ex = (
        sa.select(t0.c.p_partkey, t0.c.ps_supplycost)
        .select_from(t0)
        .where(
            t0.c.ps_supplycost
            == (
                sa.select(sa.func.min(t1.c.ps_supplycost).label("Min(ps_supplycost)"))
                .select_from(t1)
                .where(t1.c.ps_partkey == t0.c.p_partkey)
                .scalar_subquery()
            )
        )
    )

    _check(expr, ex)


def test_mutate_filter_join_no_cross_join():
    person = ibis.table(
        [('person_id', 'int64'), ('birth_datetime', 'timestamp')],
        name='person',
    )
    mutated = person.mutate(age=400)
    expr = mutated.filter(mutated.age <= 40)[mutated.person_id]

    person = sa.table(
        "person", sa.column("person_id"), sa.column("birth_datetime")
    ).alias("t1")
    t0 = sa.select(
        person.c.person_id,
        person.c.birth_datetime,
        sa.literal(400).label("age"),
    ).alias("t0")
    ex = sa.select(t0.c.person_id).where(t0.c.age <= 40)
    _check(expr, ex)


def test_filter_group_by_agg_with_same_name():
    # GH 2907
    t = ibis.table([("int_col", "int32"), ("bigint_col", "int64")], name="t")
    expr = (
        t.group_by("int_col")
        .aggregate(bigint_col=lambda t: t.bigint_col.sum())
        .filter(lambda t: t.bigint_col == 60)
    )

    t1 = sa.table("t", sa.column("int_col"), sa.column("bigint_col")).alias("t1")
    t0 = (
        sa.select(
            t1.c.int_col.label("int_col"),
            sa.func.sum(t1.c.bigint_col).label("bigint_col"),
        )
        .group_by(t1.c.int_col)
        .alias("t0")
    )
    ex = sa.select(t0).where(t0.c.bigint_col == 60)
    _check(expr, ex)


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


def test_no_cross_join(person, visited, survey):
    expr = person.join(survey, person.id == survey.person).join(
        visited,
        visited.id == survey.taken,
    )

    context = AlchemyContext(compiler=AlchemyCompiler)
    _ = AlchemyCompiler.to_sql(expr, context)

    t0 = context.get_ref(person.op())
    t1 = context.get_ref(survey.op())
    t2 = context.get_ref(visited.op())

    from_ = t0.join(t1, t0.c.id == t1.c.person).join(t2, t2.c.id == t1.c.taken)
    ex = sa.select(
        t0.c.id.label("id_x"),
        t0.c.personal,
        t0.c.family,
        t1.c.taken,
        t1.c.person,
        t1.c.quant,
        t1.c.reading,
        t2.c.id.label("id_y"),
        t2.c.site,
        t2.c.dated,
    ).select_from(from_)
    _check(expr, ex)


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


def test_gh_1045(test1, test2, test3):
    t1 = test1
    t2 = test2
    t3 = test3

    t3 = t3[[c for c in t3.columns if c != "id3"]].mutate(id3=t3.id3.cast('int64'))

    t3 = t3[[c for c in t3.columns if c != "val2"]].mutate(t3_val2=t3.id3)
    t4 = t3.join(t2, t2.id2b == t3.id3)

    t1 = t1[[t1[c].name(f"t1_{c}") for c in t1.columns]]

    expr = t1.left_join(t4, t1.t1_id1 == t4.id2a)

    test3 = sa.table("test3", sa.column("id3"), sa.column("val2"), sa.column("dt"))
    test2 = sa.table("test2", sa.column("id2a"), sa.column("id2b"), sa.column("val2"))
    test1 = sa.table("test1", sa.column("id1"), sa.column("val1"))

    t2 = test1.alias("t2")
    t0 = sa.select(
        t2.c.id1.label("t1_id1"),
        t2.c.val1.label("t1_val1"),
    ).alias("t0")

    t5 = test3.alias("t5")
    t4 = sa.select(
        t5.c.val2,
        t5.c.dt,
        sa.cast(t5.c.id3, sa.BigInteger()).label("id3"),
    ).alias("t4")
    t3 = test2.alias("t3")
    t2 = sa.select(t4.c.dt, t4.c.id3, t4.c.id3.label("t3_val2")).alias("t2")
    t1 = (
        sa.select(
            t2.c.dt,
            t2.c.id3,
            t2.c.t3_val2,
            t3.c.id2a,
            t3.c.id2b,
            t3.c.val2,
        )
        .select_from(t2.join(t3, onclause=t3.c.id2b == t2.c.id3))
        .alias("t1")
    )

    ex = sa.select(
        t0.c.t1_id1,
        t0.c.t1_val1,
        t1.c.dt,
        t1.c.id3,
        t1.c.t3_val2,
        t1.c.id2a,
        t1.c.id2b,
        t1.c.val2,
    ).select_from(t0.join(t1, isouter=True, onclause=t0.c.t1_id1 == t1.c.id2a))

    _check(expr, ex)


def test_multi_join():
    t1 = ibis.table(dict(x1="int64", y1="int64"), name="t1")
    t2 = ibis.table(dict(x2="int64"), name="t2")
    t3 = ibis.table(dict(x3="int64", y2="int64"), name="t3")
    t4 = ibis.table(dict(x4="int64"), name="t4")

    j1 = t1.join(t2, t1.x1 == t2.x2)
    j2 = t3.join(t4, t3.x3 == t4.x4)
    expr = j1.join(j2, j1.y1 == j2.y2)

    t0 = sa.table("t1", sa.column("x1"), sa.column("y1")).alias("t0")
    t1 = sa.table("t2", sa.column("x2")).alias("t1")

    t3 = sa.table("t3", sa.column("x3"), sa.column("y2")).alias("t3")
    t4 = sa.table("t4", sa.column("x4")).alias("t4")
    t2 = (
        sa.select(t3.c.x3, t3.c.y2, t4.c.x4)
        .select_from(t3.join(t4, onclause=t3.c.x3 == t4.c.x4))
        .alias("t2")
    )
    ex = sa.select(t0.c.x1, t0.c.y1, t1.c.x2, t2.c.x3, t2.c.y2, t2.c.x4,).select_from(
        t0.join(t1, onclause=t0.c.x1 == t1.c.x2).join(t2, onclause=t0.c.y1 == t2.c.y2)
    )
    _check(expr, ex)


@pytest.fixture
def h11():
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
    return q


def test_tpc_h11(h11):
    NATION = "GERMANY"
    FRACTION = 0.0001

    partsupp = sa.table(
        "partsupp",
        sa.column("ps_partkey"),
        sa.column("ps_supplycost"),
        sa.column("ps_availqty"),
        sa.column("ps_suppkey"),
    )
    supplier = sa.table(
        "supplier",
        sa.column("s_suppkey"),
        sa.column("s_nationkey"),
    )
    nation = sa.table("nation", sa.column("n_name"), sa.column("n_nationkey"))

    t4 = nation.alias("t4")
    t2 = partsupp.alias("t2")
    t3 = supplier.alias("t3")
    t1 = (
        sa.select(
            t2.c.ps_partkey,
            sa.func.sum(t2.c.ps_supplycost * t2.c.ps_availqty).label("value"),
        )
        .select_from(
            t2.join(t3, onclause=t2.c.ps_suppkey == t3.c.s_suppkey).join(
                t4, onclause=t4.c.n_nationkey == t3.c.s_nationkey
            )
        )
        .where(t4.c.n_name == NATION)
        .group_by(t2.c.ps_partkey)
    ).alias("t1")

    anon_1 = (
        sa.select(sa.func.sum(t2.c.ps_supplycost * t2.c.ps_availqty).label("total"))
        .select_from(
            t2.join(t3, onclause=t2.c.ps_suppkey == t3.c.s_suppkey).join(
                t4, onclause=t4.c.n_nationkey == t3.c.s_nationkey
            )
        )
        .where(t4.c.n_name == NATION)
        .alias("anon_1")
    )

    t0 = (
        sa.select(t1.c.ps_partkey.label("ps_partkey"), t1.c.value.label("value")).where(
            t1.c.value > sa.select(anon_1.c.total).scalar_subquery() * FRACTION
        )
    ).alias("t0")

    ex = sa.select(t0.c.ps_partkey, t0.c.value).order_by(t0.c.value.desc())
    _check(h11, ex)


def test_to_sqla_type_array_of_non_primitive():
    result = to_sqla_type(DefaultDialect(), dt.Array(dt.Struct(dict(a="int"))))
    [(result_name, result_type)] = result.value_type.pairs
    expected_name = "a"
    expected_type = sa.BigInteger()
    assert result_name == expected_name
    assert type(result_type) == type(expected_type)
    assert isinstance(result, ArrayType)


def test_no_cart_join(con, snapshot):
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
        product_level_name=lambda t: ibis.literal('-')
        .lpad(((t.ancestor_level_number - 1) * 7), '-')
        .concat(t.ancestor_level_name)
    )

    predicate = facts.product_id == products.descendant_node_natural_key
    joined = facts.join(products, predicate)

    gb = joined.group_by(products.ancestor_node_sort_order)
    agg = gb.aggregate(n=ibis.literal(1))
    ob = agg.order_by(products.ancestor_node_sort_order)

    out = str(con.compile(ob).compile(compile_kwargs=dict(literal_binds=True)))
    snapshot.assert_match(out, "out.sql")


def test_order_by_expr(con, snapshot):
    from ibis import _

    t = ibis.table(dict(a="int", b="string"), name="t")
    expr = t[_.a == 1].order_by(_.b + "a")
    out = str(con.compile(expr).compile(compile_kwargs=dict(literal_binds=True)))
    snapshot.assert_match(out, "out.sql")
