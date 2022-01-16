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
import sqlalchemy.sql as sql
from sqlalchemy import func as F
from sqlalchemy import types as sat

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyContext,
    schema_from_table,
)
from ibis.tests.expr.mocks import MockAlchemyBackend
from ibis.tests.util import assert_equal

sa = pytest.importorskip('sqlalchemy')


L = sa.literal


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
        ('real', sat.REAL, True, dt.float),
        ('bool', sat.Boolean, True, dt.boolean),
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
    expr = op(functional_alltypes.double_col, 5)
    expected = sa.select(
        [op(sa_functional_alltypes.c.double_col, L(5)).label("tmp")]
    )
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
    expr = expr_fn(functional_alltypes.double_col)
    expected = sa.select(
        [expr_fn(sa_functional_alltypes.c.double_col).label("tmp")]
    )
    _check(expr, expected)


def test_between(con, functional_alltypes, sa_functional_alltypes):
    expr = functional_alltypes.double_col.between(5, 10)
    expected = sa.select(
        [sa_functional_alltypes.c.double_col.between(L(5), L(10)).label("tmp")]
    )
    _check(expr, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        (lambda d: d.isnull(), lambda sd: sd.is_(sa.null())),
        (lambda d: d.notnull(), lambda sd: sd.isnot(sa.null())),
    ],
)
def test_isnull_notnull(
    con,
    sa_functional_alltypes,
    functional_alltypes,
    expr_fn,
    expected_fn,
):
    expr = expr_fn(functional_alltypes.double_col)
    expected = sa.select(
        [expected_fn(sa_functional_alltypes.c.double_col).label("tmp")]
    )
    _check(expr, expected)


def test_negate(sa_functional_alltypes, functional_alltypes):
    expr = -(functional_alltypes.double_col > 0)
    expected = sa.select(
        [sql.not_(sa_functional_alltypes.c.double_col > L(0)).label("tmp")]
    )
    _check(expr, expected)


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

    expr = ibis.coalesce(v2, v1, v3)
    expected = sa.select(
        [
            sa.func.coalesce(
                sa.case([(sd > L(30), sd)], else_=null),
                null,
                sf,
            ).label("tmp")
        ]
    )
    _check(expr, expected)


def test_named_expr(sa_functional_alltypes, functional_alltypes):
    expr = functional_alltypes[
        (functional_alltypes.double_col * 2).name('foo')
    ]
    expected = sa.select(
        [(sa_functional_alltypes.c.double_col * L(2)).label('foo')]
    )
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
            lambda r, n: r.inner_join(
                n, r.r_regionkey == n.n_regionkey
            ).projection(n),
            lambda rt, nt: sa.select([nt]).select_from(
                rt.join(nt, rt.c.r_regionkey == nt.c.n_regionkey)
            ),
        ),
        (
            lambda r, n: r.left_join(
                n, r.r_regionkey == n.n_regionkey
            ).projection(n),
            lambda rt, nt: sa.select([nt]).select_from(
                rt.join(
                    nt,
                    rt.c.r_regionkey == nt.c.n_regionkey,
                    isouter=True,
                )
            ),
        ),
        (
            lambda r, n: r.outer_join(
                n, r.r_regionkey == n.n_regionkey
            ).projection(n),
            lambda rt, nt: sa.select([nt]).select_from(
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


def test_join_just_materialized(con, join_just_materialized):
    joined = join_just_materialized

    nt, rt, ct = (
        con.meta.tables[name].alias(f"t{i:d}")
        for i, name in enumerate(
            ['tpch_nation', 'tpch_region', 'tpch_customer']
        )
    )

    sqla_joined = nt.join(rt, nt.c.n_regionkey == rt.c.r_regionkey).join(
        ct, nt.c.n_nationkey == ct.c.c_nationkey
    )

    expected = sa.select([sqla_joined])

    _check(joined, expected)


def test_full_outer_join(con):
    """Testing full outer join separately due to previous issue with
    outer join resulting in left outer join (issue #1773)"""
    region = con.table('tpch_region')
    nation = con.table('tpch_nation')

    predicate = region.r_regionkey == nation.n_regionkey
    joined = region.outer_join(nation, predicate)
    joined_sql_str = str(joined.compile())
    assert 'full' in joined_sql_str.lower()
    assert 'left' not in joined_sql_str.lower()


def test_simple_case(sa_alltypes, alltypes, simple_case):
    st = sa_alltypes
    expr = simple_case
    expected = sa.select(
        [
            sa.case(
                [
                    (st.c.g == L('foo'), L('bar')),
                    (st.c.g == L('baz'), L('qux')),
                ],
                else_='default',
            ).label("tmp")
        ]
    )
    _check(expr, expected)


def test_searched_case(sa_alltypes, alltypes, search_case):
    st = sa_alltypes.alias("t0")
    expr = search_case
    expected = sa.select(
        [
            sa.case(
                [
                    (st.c.f > L(0), st.c.d * L(2)),
                    (st.c.c < L(0), st.c.a * L(2)),
                ],
                else_=sa.cast(sa.null(), sa.BIGINT),
            ).label("tmp")
        ]
    )
    _check(expr, expected)


def test_where_simple_comparisons(sa_star1, where_simple_comparisons):
    expr = where_simple_comparisons
    st = sa_star1.alias("t0")
    expected = sa.select([st]).where(
        sql.and_(st.c.f > L(0), st.c.c < (st.c.f * L(2)))
    )
    _check(expr, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        (
            lambda t: t.aggregate([t['f'].sum().name('total')], [t['foo_id']]),
            lambda st: sa.select(
                [st.c.foo_id, F.sum(st.c.f).label('total')]
            ).group_by(st.c.foo_id),
        ),
        (
            lambda t: t.aggregate(
                [t['f'].sum().name('total')], ['foo_id', 'bar_id']
            ),
            lambda st: sa.select(
                [st.c.foo_id, st.c.bar_id, F.sum(st.c.f).label('total')]
            ).group_by(st.c.foo_id, st.c.bar_id),
        ),
        (
            lambda t: t.aggregate(
                [t.f.sum().name("total")],
                by=["foo_id"],
                having=[t.f.sum() > 10],
            ),
            lambda st: (
                sa.select([st.c.foo_id, F.sum(st.c.f).label("total")])
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
                sa.select([st.c.foo_id, F.sum(st.c.f).label("total")])
                .group_by(st.c.foo_id)
                .having(F.count("*") > L(100))
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
            lambda t: t.sort_by("f"),
            lambda b: sa.select([b]).order_by(b.c.f),
            id="sort_by",
        ),
        pytest.param(
            lambda t: t.sort_by(("f", 0)),
            lambda b: sa.select([b]).order_by(b.c.f.desc()),
            id="sort_by_ascending",
        ),
        pytest.param(
            lambda t: t.sort_by(["c", ("f", 0)]),
            lambda b: sa.select([b]).order_by(b.c.c, b.c.f.desc()),
            id="sort_by_mixed",
        ),
    ],
)
def test_sort_by(con, star1, sa_star1, expr_fn, expected_fn):
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
def test_limit(con, star1, sa_star1, expr_fn, expected_fn):
    expr = expr_fn(star1)
    expected = expected_fn(sa.select([sa_star1.alias("t0")]))
    _check(expr, expected)


def test_limit_filter(con, star1, sa_star1):
    expr = star1[star1.f > 0].limit(10)
    expected = sa_star1.alias("t0")
    expected = sa.select([expected]).where(expected.c.f > L(0)).limit(10)
    _check(expr, expected)


def test_limit_subquery(con, star1, sa_star1):
    expr = star1.limit(10)[lambda x: x.f > 0]
    expected = sa.select([sa_star1.alias("t1")]).limit(10).alias("t0")
    expected = sa.select([expected]).where(expected.c.f > 0)
    _check(expr, expected)


def test_cte_factor_distinct_but_equal(
    con,
    cte_factor_distinct_but_equal,
    sa_alltypes,
):
    expr = cte_factor_distinct_but_equal

    alltypes = sa_alltypes

    t2 = alltypes.alias('t2')
    t0 = (
        sa.select([t2.c.g, F.sum(t2.c.f).label('metric')])
        .group_by(t2.c.g)
        .cte('t0')
    )

    t1 = t0.alias('t1')
    table_set = t0.join(t1, t0.c.g == t1.c.g)
    stmt = sa.select([t0]).select_from(table_set)

    _check(expr, stmt)


def test_self_reference_join(con, self_reference_join, sa_star1):
    t0 = sa_star1.alias('t0')
    t1 = sa_star1.alias('t1')

    case = self_reference_join

    table_set = t0.join(t1, t0.c.foo_id == t1.c.bar_id)
    expected = sa.select([t0]).select_from(table_set)
    _check(case, expected)


def test_self_reference_in_not_exists(
    con,
    sa_functional_alltypes,
    self_reference_in_exists,
):
    semi, anti = self_reference_in_exists

    s1 = sa_functional_alltypes.alias('t0')
    s2 = sa_functional_alltypes.alias('t1')

    cond = (
        sa.exists([L(1)])
        .select_from(s1)
        .where(s1.c.string_col == s2.c.string_col)
    )

    ex_semi = sa.select([s1]).where(cond)
    ex_anti = sa.select([s1]).where(~cond)

    _check(semi, ex_semi)
    _check(anti, ex_anti)


def test_where_uncorrelated_subquery(
    con, where_uncorrelated_subquery, foo, bar
):
    expr = where_uncorrelated_subquery

    foo = con.meta.tables["foo"].alias("t0")
    bar = con.meta.tables["bar"]

    subq = sa.select([bar.c.job])
    stmt = sa.select([foo]).where(foo.c.job.in_(subq))
    _check(expr, stmt)


def test_where_correlated_subquery(con, where_correlated_subquery, foo):
    expr = where_correlated_subquery

    foo = con.meta.tables["foo"]
    t0 = foo.alias('t0')
    t1 = foo.alias('t1')
    subq = sa.select([F.avg(t1.c.y).label('mean')]).where(
        t0.c.dept_id == t1.c.dept_id
    )
    # For versions of SQLAlchemy where scalar_subquery exists,
    # it should be used (otherwise, a deprecation warning is raised)
    if hasattr(subq, 'scalar_subquery'):
        subq = subq.scalar_subquery()
    stmt = sa.select([t0]).where(t0.c.y > subq)
    _check(expr, stmt)


def test_subquery_aliased(con, subquery_aliased, star1, star2):
    expr = subquery_aliased

    s1 = con.meta.tables["star1"].alias("t2")
    s2 = con.meta.tables["star2"].alias("t1")

    agged = (
        sa.select([s1.c.foo_id, F.sum(s1.c.f).label('total')])
        .group_by(s1.c.foo_id)
        .alias('t0')
    )

    joined = agged.join(s2, agged.c.foo_id == s2.c.foo_id)
    expected = sa.select([agged, s2.c.value1]).select_from(joined)

    _check(expr, expected)


def test_lower_projection_sort_key(con, subquery_aliased, star1, star2):
    expr = subquery_aliased

    t3 = con.meta.tables["star1"].alias("t3")
    t2 = con.meta.tables["star2"].alias("t2")

    t4 = (
        sa.select([t3.c.foo_id, F.sum(t3.c.f).label('total')])
        .group_by(t3.c.foo_id)
        .alias('t4')
    )
    t1 = (
        sa.select([t4.c.foo_id, t4.c.total, t2.c.value1])
        .select_from(t4.join(t2, t4.c.foo_id == t2.c.foo_id))
        .alias('t1')
    )
    t0 = (
        sa.select([t1.c.foo_id, t1.c.total, t1.c.value1])
        .where(t1.c.total > L(100))
        .alias('t0')
    )
    expected = sa.select([t0.c.foo_id, t0.c.total, t0.c.value1]).order_by(
        t0.c.total.desc()
    )

    expr2 = expr[expr.total > 100].sort_by(ibis.desc('total'))
    _check(expr2, expected)


def test_exists(con, exists, t1, t2):
    e1, e2 = exists

    t1 = con.meta.tables["foo_t"].alias("t0")
    t2 = con.meta.tables["bar_t"].alias("t1")

    cond1 = sa.exists([L(1)]).where(t1.c.key1 == t2.c.key1)
    ex1 = sa.select([t1]).where(cond1)

    cond2 = sa.exists([L(1)]).where(
        sql.and_(t1.c.key1 == t2.c.key1, t2.c.key2 == L('foo'))
    )
    ex2 = sa.select([t1]).where(cond2)

    _check(e1, ex1)
    _check(e2, ex2)


def test_not_exists(con, not_exists):
    t1 = con.table("t1")
    t2 = con.table("t2")

    expr = not_exists

    t1 = con.meta.tables["foo_t"].alias("t0")
    t2 = con.meta.tables["bar_t"].alias("t1")

    expected = sa.select([t1]).where(
        sa.not_(sa.exists([L(1)]).where(t1.c.key1 == t2.c.key1))
    )

    _check(expr, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        (lambda t: t.distinct(), lambda sat: sa.select([sat]).distinct()),
        (
            lambda t: t['string_col', 'int_col'].distinct(),
            lambda sat: sa.select(
                [sat.c.string_col, sat.c.int_col]
            ).distinct(),
        ),
        (
            lambda t: t.string_col.distinct(),
            lambda sat: sa.select([sat.c.string_col.distinct()]),
        ),
        (
            lambda t: t.int_col.nunique().name('nunique'),
            lambda sat: sa.select(
                [F.count(sat.c.int_col.distinct()).label('nunique')]
            ),
        ),
        (
            lambda t: t.group_by('string_col').aggregate(
                t.int_col.nunique().name('nunique')
            ),
            lambda sat: sa.select(
                [
                    sat.c.string_col,
                    F.count(sat.c.int_col.distinct()).label('nunique'),
                ]
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
    expr = agg.sort_by(ibis.desc('foo'))

    sat = sa_functional_alltypes.alias("t1")
    base = (
        sa.select(
            [sat.c.string_col, F.max(sat.c.double_col).label('foo')]
        ).group_by(sat.c.string_col)
    ).alias('t0')

    ex = (
        sa.select([base.c.string_col, base.c.foo])
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
        sa.select([part_t1.c.p_partkey, partsupp_t2.c.ps_supplycost])
        .select_from(
            part_t1.join(
                partsupp_t2,
                onclause=part_t1.c.p_partkey == partsupp_t2.c.ps_partkey,
            )
        )
        .alias("t0")
    )

    partsupp_t2 = partsupp.alias("t2")
    supplier_t5 = supplier.alias("t5")
    t3 = (
        sa.select([partsupp_t2.c.ps_partkey, partsupp_t2.c.ps_supplycost])
        .select_from(
            partsupp_t2.join(
                supplier_t5,
                onclause=supplier_t5.c.s_suppkey == partsupp_t2.c.ps_suppkey,
            )
        )
        .alias("t3")
    )
    ex = (
        sa.select([t0.c.p_partkey, t0.c.ps_supplycost])
        .select_from(t0)
        .where(
            t0.c.ps_supplycost
            == (
                sa.select([sa.func.min(t3.c.ps_supplycost).label("min")])
                .select_from(t3)
                .where(t3.c.ps_partkey == t0.c.p_partkey)
                .as_scalar()
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
        [
            person.c.person_id,
            person.c.birth_datetime,
            sa.literal(400).label("age"),
        ]
    ).alias("t0")
    ex = sa.select([t0.c.person_id]).where(t0.c.age <= 40)
    _check(expr, ex)


def test_filter_group_by_agg_with_same_name():
    # GH 2907
    t = ibis.table([("int_col", "int32"), ("bigint_col", "int64")], name="t")
    expr = (
        t.group_by("int_col")
        .aggregate(bigint_col=lambda t: t.bigint_col.sum())
        .filter(lambda t: t.bigint_col == 60)
    )

    t1 = sa.table("t", sa.column("int_col"), sa.column("bigint_col")).alias(
        "t1"
    )
    t0 = (
        sa.select(
            [
                t1.c.int_col.label("int_col"),
                sa.func.sum(t1.c.bigint_col).label("bigint_col"),
            ]
        )
        .group_by(t1.c.int_col)
        .alias("t0")
    )
    ex = sa.select([t0]).where(t0.c.bigint_col == 60)
    _check(expr, ex)
