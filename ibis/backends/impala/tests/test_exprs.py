from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.types as ir
from ibis import literal as L
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.expr import api


def test_embedded_identifier_quoting(alltypes):
    t = alltypes

    expr = t[[(t.double_col * 2).name("double(fun)")]]["double(fun)"].sum()
    expr.execute()


def test_decimal_metadata(con):
    table = con.table("lineitem")

    expr = table.l_quantity
    assert expr.type().precision == 12
    assert expr.type().scale == 2

    # TODO: what if user impyla version does not have decimal Metadata?


def test_builtins(con, alltypes):
    table = alltypes

    i1 = table.tinyint_col
    i4 = table.int_col
    i8 = table.bigint_col
    d = table.double_col
    s = table.string_col

    exprs = [
        api.now(),
        api.e,
        # hash functions
        i4.hash(),
        d.hash(),
        s.hash(),
        # modulus cases
        i1 % 5,
        i4 % 10,
        20 % i1,
        d % 5,
        pytest.warns(FutureWarning, i1.zeroifnull),
        pytest.warns(FutureWarning, i4.zeroifnull),
        pytest.warns(FutureWarning, i8.zeroifnull),
        i4.to_timestamp("s"),
        i4.to_timestamp("ms"),
        i4.to_timestamp("us"),
        i8.to_timestamp(),
        d.abs(),
        d.cast("decimal(12, 2)"),
        d.cast("int32"),
        d.ceil(),
        d.exp(),
        d.isnull(),
        d.fillna(0),
        d.floor(),
        d.log(),
        d.ln(),
        d.log2(),
        d.log10(),
        d.notnull(),
        pytest.warns(FutureWarning, d.zeroifnull),
        pytest.warns(FutureWarning, d.nullifzero),
        d.round(),
        d.round(2),
        d.round(i1),
        i1.sign(),
        i4.sign(),
        d.sign(),
        # conv
        i1.convert_base(10, 2),
        i4.convert_base(10, 2),
        i8.convert_base(10, 2),
        s.convert_base(10, 2),
        d.sqrt(),
        pytest.warns(FutureWarning, d.zeroifnull),
        # nullif cases
        5 / i1.nullif(0),
        5 / i1.nullif(i4),
        5 / i4.nullif(0),
        5 / d.nullif(0),
        api.literal(5).isin([i1, i4, d]),
        # tier and histogram
        d.bucket([0, 10, 25, 50, 100]),
        d.bucket([0, 10, 25, 50], include_over=True),
        d.bucket([0, 10, 25, 50], include_over=True, close_extreme=False),
        d.bucket([10, 25, 50, 100], include_under=True),
        d.histogram(10),
        d.histogram(5, base=10),
        d.histogram(base=10, binwidth=5),
        # coalesce-like cases
        api.coalesce(
            table.int_col, api.null(), table.smallint_col, table.bigint_col, 5
        ),
        api.greatest(table.float_col, table.double_col, 5),
        api.least(table.string_col, "foo"),
        # string stuff
        s.contains("6"),
        s.like("6%"),
        s.re_search(r"[\d]+"),
        s.re_extract(r"[\d]+", 0),
        s.re_replace(r"[\d]+", "a"),
        s.repeat(2),
        s.translate("a", "b"),
        s.find("a"),
        s.lpad(10, "a"),
        s.rpad(10, "a"),
        s.find_in_set(["a"]),
        s.lower(),
        s.upper(),
        s.reverse(),
        s.ascii_str(),
        s.length(),
        s.strip(),
        s.lstrip(),
        s.strip(),
        # strings with int expr inputs
        s.left(i1),
        s.right(i1),
        s.substr(i1, i1 + 2),
        s.repeat(i1),
    ]

    proj_exprs = [expr.name("e%d" % i) for i, expr in enumerate(exprs)]

    projection = table[proj_exprs]
    projection.limit(10).execute()

    _check_impala_output_types_match(con, projection)


def _check_impala_output_types_match(con, table):
    query = ImpalaCompiler.to_sql(table)
    t = con.sql(query)

    left_schema, right_schema = t.schema(), table.schema()
    for n, left_ty, right_ty in zip(
        left_schema.names, left_schema.types, right_schema.types
    ):
        assert (
            left_ty == right_ty
        ), f"Value for {n} had left type {left_ty} and right type {right_ty}\nquery:\n{query}"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # mod cases
        (L(50) % 5, 0),
        (L(50000) % 10, 0),
        (250 % L(50), 0),
        # nullif cases
        (5 / L(50).nullif(0), 0.1),
        (5 / L(50).nullif(L(50000)), 0.1),
        (5 / L(50000).nullif(0), 0.0001),
        (L(50000).fillna(0), 50000),
    ],
)
def test_int_builtins(con, expr, expected):
    result = con.execute(expr)
    assert result == expected, ImpalaCompiler.to_sql(expr)


@pytest.mark.parametrize(
    ("col", "expected"),
    [
        param("tinyint_col", "int8", id="tinyint"),
        param("smallint_col", "int16", id="smallint"),
        param("int_col", "int32", id="int"),
        param("bigint_col", "int64", id="bigint"),
        param("float_col", "float32", id="float"),
        param("double_col", "float64", id="double"),
        param("timestamp_col", "datetime64[ns]", id="timestamp"),
    ],
)
def test_column_types(alltypes_df, col, expected):
    assert alltypes_df[col].dtype.name == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L(50000).to_timestamp("s"), pd.to_datetime(50000, unit="s")),
        (L(50000).to_timestamp("ms"), pd.to_datetime(50000, unit="ms")),
        (L(5 * 10**8).to_timestamp(), pd.to_datetime(5 * 10**8, unit="s")),
        (
            ibis.timestamp("2009-05-17 12:34:56").truncate("y"),
            pd.Timestamp("2009-01-01"),
        ),
        (
            ibis.timestamp("2009-05-17 12:34:56").truncate("M"),
            pd.Timestamp("2009-05-01"),
        ),
        (
            ibis.timestamp("2009-05-17 12:34:56").truncate("month"),
            pd.Timestamp("2009-05-01"),
        ),
        (
            ibis.timestamp("2009-05-17 12:34:56").truncate("d"),
            pd.Timestamp("2009-05-17"),
        ),
        (
            ibis.timestamp("2009-05-17 12:34:56").truncate("h"),
            pd.Timestamp("2009-05-17 12:00"),
        ),
        (
            ibis.timestamp("2009-05-17 12:34:56").truncate("m"),
            pd.Timestamp("2009-05-17 12:34"),
        ),
        (
            ibis.timestamp("2009-05-17 12:34:56").truncate("minute"),
            pd.Timestamp("2009-05-17 12:34"),
        ),
    ],
)
def test_timestamp_builtins(con, expr, expected):
    result = con.execute(expr)
    assert result == expected, ImpalaCompiler.to_sql(expr)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L(-5).abs(), 5),
        (L(5.245).cast("int32"), 5),
        (L(5.245).ceil(), 6),
        (L(5.245).isnull(), False),
        (L(5.245).floor(), 5),
        (L(5.245).notnull(), True),
        (L(5.245).round(), 5),
        (L(5.245).round(2), Decimal("5.25")),
        (L(5.245).sign(), 1),
    ],
)
def test_decimal_builtins(con, expr, expected):
    result = con.execute(expr)
    assert result == expected, ImpalaCompiler.to_sql(expr)


def approx_equal(a, b, eps):
    assert abs(a - b) < eps


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        pytest.param(lambda dc: dc, "5.245", id="id"),
        pytest.param(lambda dc: dc % 5, "0.245", id="mod"),
        pytest.param(lambda dc: dc.fillna(0), "5.245", id="fillna"),
        pytest.param(lambda dc: dc.exp(), "189.6158", id="exp"),
        pytest.param(lambda dc: dc.log(), "1.65728", id="log"),
        pytest.param(lambda dc: dc.log2(), "2.39094", id="log2"),
        pytest.param(lambda dc: dc.log10(), "0.71975", id="log10"),
        pytest.param(lambda dc: dc.sqrt(), "2.29019", id="sqrt"),
        pytest.param(
            lambda dc: pytest.warns(FutureWarning, dc.zeroifnull),
            "5.245",
            id="zeroifnull",
        ),
        pytest.param(lambda dc: -dc, "-5.245", id="neg"),
    ],
)
def test_decimal_builtins_2(con, func, expected):
    dc = L("5.245").cast("decimal(12, 5)")
    expr = func(dc)
    result = con.execute(expr)
    tol = Decimal("0.0001")
    approx_equal(Decimal(result), Decimal(expected), tol)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("abcd").length(), 4),
        (L("ABCD").lower(), "abcd"),
        (L("abcd").upper(), "ABCD"),
        (L("abcd").reverse(), "dcba"),
        (L("abcd").ascii_str(), 97),
        (L("   a   ").strip(), "a"),
        (L("   a   ").lstrip(), "a   "),
        (L("   a   ").rstrip(), "   a"),
        (L("abcd").capitalize(), "Abcd"),
        (L("abcd").substr(0, 2), "ab"),
        (L("abcd").left(2), "ab"),
        (L("abcd").right(2), "cd"),
        (L("abcd").repeat(2), "abcdabcd"),
        # global replace not available in Impala yet
        # (L('aabbaabbaa').replace('bb', 'B'), 'aaBaaBaa'),
        (L("0123").translate("012", "abc"), "abc3"),
        (L("abcd").find("a"), 0),
        (L("baaaab").find("b", 2), 5),
        (L("abcd").lpad(1, "-"), "a"),
        (L("abcd").lpad(5), " abcd"),
        (L("abcd").rpad(1, "-"), "a"),
        (L("abcd").rpad(5), "abcd "),
        (L("abcd").find_in_set(["a", "b", "abcd"]), 2),
        (L(", ").join(["a", "b"]), "a, b"),
        (L("abcd").like("a%"), True),
        (L("abcd").re_search("[a-z]"), True),
        (L("abcd").re_extract("[a-z]", 0), "a"),
        (L("abcd").re_replace("(b)", "2"), "a2cd"),
    ],
)
def test_string_functions(con, expr, expected):
    result = con.execute(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("https://www.cloudera.com").host(), "www.cloudera.com"),
        (
            L("https://www.youtube.com/watch?v=kEuEcWfewf8&t=10").query("v"),
            "kEuEcWfewf8",
        ),
    ],
)
def test_parse_url(con, expr, expected):
    result = con.execute(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L(7) / 2, 3.5),
        (L(7) // 2, 3),
        (L(7).floordiv(2), 3),
        (L(2).rfloordiv(7), 3),
    ],
)
def test_div_floordiv(con, expr, expected):
    result = con.execute(expr)
    assert result == expected


def test_filter_predicates(con):
    t = con.table("nation")

    predicates = [
        lambda x: x.n_name.lower().like("%ge%"),
        lambda x: x.n_name.lower().contains("ge"),
        lambda x: x.n_name.lower().rlike(".*ge.*"),
    ]

    expr = t
    for pred in predicates:
        expr = expr[pred(expr)].select(expr)

    expr.execute()


def test_histogram_value_counts(alltypes):
    t = alltypes
    expr = t.double_col.histogram(10).value_counts()
    expr.execute()


def test_casted_expr_impala_bug(alltypes):
    # Per GH #396. Prior to Impala 2.3.0, there was a bug in the query
    # planner that caused this expression to fail
    expr = alltypes.string_col.cast("double").value_counts()
    expr.execute()


def test_decimal_timestamp_builtins(con):
    table = con.table("lineitem")

    dc = table.l_quantity
    ts = table.l_receiptdate.cast("timestamp")

    exprs = [
        dc % 10,
        dc + 5,
        dc + dc,
        dc / 2,
        dc * 2,
        dc**2,
        dc.cast("double"),
        api.ifelse(table.l_discount > 0, dc * table.l_discount, api.NA),
        dc.fillna(0),
        ts < (ibis.now() + ibis.interval(months=3)),
        ts < (ibis.timestamp("2005-01-01") + ibis.interval(months=3)),
        # hashing
        dc.hash(),
        ts.hash(),
        # truncate
        ts.truncate("y"),
        ts.truncate("q"),
        ts.truncate("month"),
        ts.truncate("d"),
        ts.truncate("w"),
        ts.truncate("h"),
        ts.truncate("minute"),
    ]

    timestamp_fields = [
        "years",
        "months",
        "days",
        "hours",
        "minutes",
        "seconds",
        "weeks",
    ]
    for field in timestamp_fields:
        if hasattr(ts, field):
            exprs.append(getattr(ts, field)())

        offset = ibis.interval(**{field: 2})
        exprs.append(ts + offset)
        exprs.append(ts - offset)

    proj_exprs = [expr.name("e%d" % i) for i, expr in enumerate(exprs)]

    projection = table[proj_exprs].limit(10)
    projection.execute()


def test_timestamp_scalar_in_filter(alltypes):
    # #310
    table = alltypes

    expr = table.filter(
        [
            table.timestamp_col
            < (ibis.timestamp("2010-01-01") + ibis.interval(months=3)),
            table.timestamp_col < (ibis.now() + ibis.interval(days=10)),
        ]
    ).count()
    expr.execute()


def test_aggregations(alltypes):
    table = alltypes.limit(100)

    d = table.double_col
    s = table.string_col

    cond = table.string_col.isin(["1", "7"])

    exprs = [
        table.bool_col.count(),
        d.sum(),
        d.mean(),
        d.min(),
        d.max(),
        s.approx_nunique(),
        d.approx_median(),
        s.group_concat(),
        d.std(),
        d.std(how="pop"),
        d.var(),
        d.var(how="pop"),
        table.bool_col.any(),
        table.bool_col.notany(),
        -table.bool_col.any(),
        table.bool_col.all(),
        table.bool_col.notall(),
        -table.bool_col.all(),
        table.bool_col.count(where=cond),
        d.sum(where=cond),
        d.mean(where=cond),
        d.min(where=cond),
        d.max(where=cond),
        d.std(where=cond),
        d.var(where=cond),
    ]

    metrics = [expr.name("e%d" % i) for i, expr in enumerate(exprs)]

    agged_table = table.aggregate(metrics)
    agged_table.execute()


def test_analytic_functions(alltypes):
    t = alltypes.limit(1000)

    g = t.group_by("string_col").order_by("double_col")
    f = t.float_col

    exprs = [
        f.lag(),
        f.lead(),
        f.rank(),
        f.dense_rank(),
        f.percent_rank(),
        f.ntile(buckets=7),
        f.first(),
        f.last(),
        f.first().over(ibis.window(preceding=10)),
        f.first().over(ibis.window(following=10)),
        ibis.row_number(),
        f.cumsum(),
        f.cummean(),
        f.cummin(),
        f.cummax(),
        # boolean cumulative reductions
        (f == 0).cumany(),
        (f == 0).cumall(),
        f.sum(),
        f.mean(),
        f.min(),
        f.max(),
    ]

    proj_exprs = [expr.name("e%d" % i) for i, expr in enumerate(exprs)]

    proj_table = g.mutate(proj_exprs)
    proj_table.execute()


def test_anti_join_self_reference_works(con, alltypes):
    t = alltypes.limit(100)
    t2 = t.view()
    case = t[-((t.string_col == t2.string_col).any())]
    con.explain(case)


def test_tpch_self_join_failure(con):
    region = con.table("region")
    nation = con.table("nation")
    customer = con.table("customer")
    orders = con.table("orders")

    fields_of_interest = [
        region.r_name.name("region"),
        nation.n_name.name("nation"),
        orders.o_totalprice.name("amount"),
        orders.o_orderdate.cast("timestamp").name("odate"),
    ]

    joined_all = (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)[fields_of_interest]
    )

    year = joined_all.odate.year().name("year")
    total = joined_all.amount.sum().cast("double").name("total")
    annual_amounts = joined_all.group_by(["region", year]).aggregate(total)

    current = annual_amounts
    prior = annual_amounts.view()

    yoy_change = (current.total - prior.total).name("yoy_change")
    yoy = current.join(
        prior,
        ((current.region == prior.region) & (current.year == (prior.year - 1))),
    )[current.region, current.year, yoy_change]

    # no analysis failure
    con.explain(yoy)


def test_tpch_correlated_subquery_failure(con):
    # #183 and other issues
    region = con.table("region")
    nation = con.table("nation")
    customer = con.table("customer")
    orders = con.table("orders")

    fields_of_interest = [
        customer,
        region.r_name.name("region"),
        orders.o_totalprice.name("amount"),
        orders.o_orderdate.cast("timestamp").name("odate"),
    ]

    tpch = (
        region.join(nation, region.r_regionkey == nation.n_regionkey)
        .join(customer, customer.c_nationkey == nation.n_nationkey)
        .join(orders, orders.o_custkey == customer.c_custkey)[fields_of_interest]
    )

    t2 = tpch.view()
    conditional_avg = t2[(t2.region == tpch.region)].amount.mean()
    amount_filter = tpch.amount > conditional_avg

    expr = tpch[amount_filter].limit(0)
    con.explain(expr)


def test_non_equijoin(con):
    t = con.table("functional_alltypes").limit(100)
    t2 = t.view()

    expr = t.join(t2, t.tinyint_col < t2.timestamp_col.minute()).count()

    # it works
    expr.execute()


def test_char_varchar_types(con):
    sql = """\
SELECT CAST(string_col AS varchar(20)) AS varchar_col,
   CAST(string_col AS CHAR(5)) AS char_col
FROM functional_alltypes"""

    t = con.sql(sql)

    assert isinstance(t.varchar_col, ir.StringColumn)
    assert isinstance(t.char_col, ir.StringColumn)


def test_unions_with_ctes(con, alltypes):
    t = alltypes

    expr1 = t.group_by(["tinyint_col", "string_col"]).aggregate(
        t.double_col.sum().name("metric")
    )
    expr2 = expr1.view()

    join1 = expr1.join(expr2, expr1.string_col == expr2.string_col)[[expr1]]
    join2 = join1.view()

    expr = join1.union(join2)
    con.explain(expr)


def test_head(con):
    t = con.table("functional_alltypes")
    result = t.head().execute()
    expected = t.limit(5).execute()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (ibis.NA.cast("int64"), ibis.NA.cast("int64"), True),
        (L(1), L(1), True),
        (ibis.NA.cast("int64"), L(1), False),
        (L(1), ibis.NA.cast("int64"), False),
        (L(0), L(1), False),
        (L(1), L(0), False),
    ],
)
def test_identical_to(con, left, right, expected):
    expr = left.identical_to(right)
    result = con.execute(expr)
    assert result == expected


def test_not(alltypes):
    t = alltypes.limit(10)
    expr = t.select(double_col=~t.double_col.isnull())
    result = expr.execute().double_col
    expected = ~t.execute().double_col.isnull()
    tm.assert_series_equal(result, expected)


def test_where_with_timestamp(snapshot):
    t = ibis.table(
        [("uuid", "string"), ("ts", "timestamp"), ("search_level", "int64")],
        name="t",
    )
    expr = t.group_by(t.uuid).aggregate(min_date=t.ts.min(where=t.search_level == 1))
    snapshot.assert_match(ibis.impala.compile(expr), "out.sql")


def test_filter_with_analytic(snapshot):
    x = ibis.table(ibis.schema([("col", "int32")]), "x")
    with_filter_col = x[x.columns + [ibis.null().name("filter")]]
    filtered = with_filter_col[with_filter_col["filter"].isnull()]
    subquery = filtered[filtered.columns]

    with_analytic = subquery[["col", subquery.count().name("analytic")]]
    expr = with_analytic[with_analytic.columns]

    snapshot.assert_match(ibis.impala.compile(expr), "out.sql")


def test_named_from_filter_group_by(snapshot):
    t = ibis.table([("key", "string"), ("value", "double")], name="t0")
    gb = t.filter(t.value == 42).group_by(t.key)
    sum_expr = lambda t: (t.value + 1 + 2 + 3).sum()
    expr = gb.aggregate(abc=sum_expr)
    snapshot.assert_match(ibis.impala.compile(expr), "abc.sql")

    expr = gb.aggregate(foo=sum_expr)
    snapshot.assert_match(ibis.impala.compile(expr), "foo.sql")


def test_nunique_where(snapshot):
    t = ibis.table([("key", "string"), ("value", "double")], name="t0")
    expr = t.key.nunique(where=t.value >= 1.0)
    snapshot.assert_match(ibis.impala.compile(expr), "out.sql")
