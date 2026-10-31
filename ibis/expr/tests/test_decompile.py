from __future__ import annotations

import pytest

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import _
from ibis.expr.decompile import decompile

t = ibis.table([("a", "int64"), ("b", "string")], name="t")
u = ibis.table([("a", "int64"), ("c", "float64")], name="u")
odd = ibis.table(
    [("count", "int64"), ("my col", "int64"), ("class", "int64")], name="odd"
)
expr = t.a.sum()

countries = ibis.table(
    [
        ("name", "string"),
        ("continent", "string"),
        ("population", "int"),
        ("area_km2", "int"),
    ],
    name="countries",
)
asian_countries = countries.filter(countries.continent == "AS")
eurasian_countries = countries.filter(countries.continent.isin(("AS", "EUR")))
top_with_highest_population = asian_countries.order_by(
    asian_countries.population.desc()
).limit(10)

overall_population_density = (
    asian_countries.population.sum() / asian_countries.area_km2.sum()
)
population_density_per_country = asian_countries.group_by("name").aggregate(
    _.population.sum() / _.area_km2.sum()
)

one = ibis.literal(1)
two = ibis.literal(2)
three = one + two
nine = three**2

total = t.a.sum()
shared_exists = ops.ExistsSubquery(ops.Filter(u, [u.a == t.a])).to_expr()
roundtrip_cases = {
    "logical_operators": t.filter(((t.a > 1) & ~(t.b == "x")) | (t.a < -t.a)),
    "isin_values": t.filter(t.a.isin([1, 2])),
    "typed_literal": t.select(ibis.literal(1, "int8").name("x")),
    "typed_literal_operand": t.select(
        (ibis.literal("0.2", "decimal(2, 1)") * t.a).name("x")
    ),
    "literal_only_operands": t.select(((ibis.literal(1) + 2) == 3).name("eq")),
    "interval_literal": ibis.interval(1, "D"),
    "exists": t.filter((u.a == t.a).any()),
    "not_exists": t.filter(~((u.a == t.a) & (u.c > 1)).any()),
    "exists_on_filtered": asian_countries.filter(
        (u.a == asian_countries.population).any()
    ),
    "in_subquery": t.filter(t.a.isin(u.a)),
    "aliased_unary": t.select(
        (~(t.a > 1)).name("x"), (-t.a).name("neg"), t.a.negate().negate().name("k")
    ),
    "nullif": t.select(t.b.nullif("x").name("n")),
    "as_interval": t.select(t.a.as_interval("D").name("i")),
    "difference_all": t.select(t.a).difference(u.select(u.a), distinct=False),
    "intersect_all": t.select(t.a).intersect(u.select(u.a), distinct=False),
    "union_distinct": t.select(t.a).union(u.select(u.a), distinct=True),
    "correlated_scalar_subquery": t.filter(
        t.a > u.filter(u.a == t.a).aggregate(m=u.a.mean()).as_scalar()
    ),
    "table_as_scalar": t.filter(t.a > u.aggregate(m=u.a.max()).as_scalar()),
    "window_range": t.select(
        t.a.sum().over(group_by=t.b, order_by=t.a, range=(None, 0)).name("s")
    ),
    "window_rows": t.select(t.a.sum().over(order_by=t.a, rows=(-1, 0)).name("s")),
    "window_rank": t.select(t.a.rank().name("r"), ibis.row_number().name("n")),
    "window_ntile": t.select(ibis.ntile(2).over(order_by=t.a).name("n")),
    "window_lag": t.select(t.a.lag(1, 0).over(order_by=t.a).name("p")),
    "cross_join": t.cross_join(u),
    "join_chain": t.join(u, t.a == u.a).select(t.b, u.c),
    "reduction_where": t.aggregate(s=t.a.sum(where=t.b == "x")),
    "reduction_how": t.aggregate(s=t.a.std(how="pop")),
    "aggregate_by_only": t.aggregate(by=["a"]),
    "shared_metric": t.aggregate([total.name("x"), total.name("y")]),
    "dummy_table": ibis.literal(1).name("x").as_table(),
    "coalesce": t.select(ibis.coalesce(t.b, "none").name("b")),
    "truncate": ibis.timestamp("2024-01-01 00:00:00").truncate("M"),
    "negated_literal": t.select(
        (-ibis.literal(1)).name("x"), (t.a + -ibis.literal(1)).name("y")
    ),
    "inverted_literal": t.filter(~ibis.literal(True)),
    "in_subquery_column_names": t.filter(
        t.a.isin(odd["count"]), t.a.isin(odd["my col"]), t.a.isin(odd["class"])
    ),
    "in_subquery_aggregate_column": t.filter(
        t.a.isin(u.aggregate(count=u.a.max())["count"])
    ),
    "in_subquery_renamed": t.filter(t.a.isin(u.select(x=u.a).x)),
    "scalar_subquery_column_names": t.filter(
        t.a > odd["count"].as_scalar(), t.a < odd["my col"].as_scalar()
    ),
    "window_zero_preceding": t.select(
        t.a.sum().over(ibis.window(preceding=0, following=2, order_by=t.a)).name("s")
    ),
    "window_range_zero_preceding": t.select(
        t.a.sum()
        .over(ibis.range_window(preceding=0.0, following=0, order_by=t.a))
        .name("s")
    ),
    "window_range_unbounded": t.select(
        t.a.sum().over(order_by=t.a, range=(None, None)).name("s")
    ),
    "window_lag_default": t.select(t.a.lag(default=0).over(order_by=t.a).name("p")),
    "clip_upper": t.select(t.a.clip(upper=5).name("x")),
    "null_decimal": ibis.literal(None, "decimal(2, 1)"),
    "in_subquery_derived": t.filter(t.a.isin(u.a + 1)),
    "in_subquery_derived_named": t.filter(t.a.isin((u.a + 1).name("x"))),
    "exists_with_in_subquery": t.filter(((u.a == t.a) & u.c.isin(odd["count"])).any()),
    "exists_in_reduction": t.aggregate(n=t.count(where=(u.a == t.a).any())),
    "exists_sort_key": t.order_by((u.a == t.a).any()),
    "exists_group_key": t.group_by(e=(u.a == t.a).any()).aggregate(n=t.count()),
    # EXISTS that no public method spells
    "exists_uncorrelated": t.filter(
        ops.ExistsSubquery(u.filter(u.a > 1).op()).to_expr()
    ),
    "exists_over_projection": t.filter(
        ops.ExistsSubquery(u.filter(u.a == t.a).select(u.a).op()).to_expr()
    ),
    "exists_shared_across_relations": t.filter(shared_exists).select(
        shared_exists.name("x")
    ),
}


def test_decompile_invalid_type():
    schema = ibis.schema([("a", "int64"), ("b", "string")])
    with pytest.raises(TypeError):
        decompile(schema, assign_result_to=None, render_import=False, format=False)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (top_with_highest_population, top_with_highest_population),
        (overall_population_density, overall_population_density),
        (population_density_per_country, population_density_per_country),
        (eurasian_countries, eurasian_countries),
        (three, three),
        (nine, nine),
    ],
    ids=[
        "top_with_highest_population",
        "overall_population_density",
        "population_density_per_country",
        "eurasian_countries",
        "three",
        "nine",
    ],
)
def test_basic(expr, expected):
    rendered = decompile(expr)

    locals_ = {}
    exec(rendered, {}, locals_)  # noqa: S102
    restored = locals_["result"]

    if isinstance(expected, ir.Expr):
        assert restored.equals(expected)
    else:
        assert restored == expected


# The contract: the generated code rebuilds an expression that compiles to
# equivalent SQL, and decompiling that expression gives the same code back.
# The rebuilt expression `equals` the original when the original is in the form
# the public API builds, which is what these cases are.
@pytest.mark.parametrize("expr", roundtrip_cases.values(), ids=roundtrip_cases)
def test_roundtrip(expr):
    rendered = decompile(expr)

    locals_ = {}
    exec(rendered, {}, locals_)  # noqa: S102
    restored = locals_["result"]

    assert restored.equals(expr)
    # the generated code is a fixed point
    assert decompile(restored) == rendered


def test_roundtrip_self_join():
    # a view gets a fresh identity each time, so compare the generated SQL
    expr = t.join(t.view(), "a")
    rendered = decompile(expr)

    locals_ = {}
    exec(rendered, {}, locals_)  # noqa: S102
    restored = locals_["result"]

    assert ibis.to_sql(restored) == ibis.to_sql(expr)


def test_roundtrip_exists_over_several_predicates():
    # ``.any()`` builds one predicate, so several inner ones come back as their
    # conjunction: the same SQL up to parentheses
    inner = u.filter(u.c > 1, u.a == t.a)
    expr = t.filter(ops.ExistsSubquery(inner.op()).to_expr())
    rendered = decompile(expr)

    locals_ = {}
    exec(rendered, {}, locals_)  # noqa: S102
    restored = locals_["result"]

    assert restored.equals(t.filter(((u.c > 1) & (u.a == t.a)).any()))
    assert decompile(restored) == rendered


def test_variable_names_avoid_keywords_and_builtins():
    rendered = decompile(roundtrip_cases["shared_metric"])
    assert "sum_ = t.a.sum()" in rendered
    assert "\nsum = " not in rendered


def test_view():
    expr = ibis.table({"x": "int"}, name="t").view()
    assert "t.view()" in decompile(expr)


def test_distinct():
    expr = ibis.table({"x": "int"}, name="t").distinct()
    assert "t.distinct()" in decompile(expr)


@pytest.mark.parametrize(
    ("method", "override"),
    [
        (ibis._.x.upper(), ".upper()"),
        (ibis._.x.lower(), ".lower()"),
        (ibis._.y.minute(), ".minute()"),
    ],
)
def test_method_overrides(method, override):
    expr = ibis.table({"x": "string", "y": "timestamp"}, name="t").select(method)
    assert override in decompile(expr)
