from __future__ import annotations

import re

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
from ibis import deferred as _
from ibis import selectors as s


@pytest.fixture
def t():
    return ibis.table(
        dict(
            a="int",
            b="string",
            c="array<string>",
            d="struct<a: array<map<string, array<float>>>>",
            e="float",
            f="decimal(3, 1)",
            g="array<array<map<float, float>>>",
            ga="string",
        ),
        name="t",
    )


@pytest.mark.parametrize(
    "sel",
    [s.where(lambda _: False), s.startswith("X"), s.endswith("🙂")],
    ids=["false", "startswith", "endswith"],
)
def test_empty_selection(t, sel):
    with pytest.raises(exc.IbisError):
        t.select(sel)


def test_where(t):
    assert t.select(s.where(lambda _: True)).equals(t.select(*t.columns))


def test_numeric(t):
    assert t.select(s.numeric()).equals(t.select("a", "e", "f"))


@pytest.mark.parametrize(
    ("obj", "expected"),
    [(dt.Array, ("c", "g")), ("float", ("e",)), (dt.Decimal(3, 1), ("f",))],
    ids=["type", "string", "instance"],
)
def test_of_type(t, obj, expected):
    assert t.select(s.of_type(obj)).equals(t.select(*expected))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("array", ["c_array"]),
        ("decimal", ["c_dec52"]),
        ("floating", ["c_f32", "c_f64"]),
        ("geospatial", ["c_point"]),
        ("integer", ["c_i32", "c_u64"]),
        ("map", ["c_map"]),
        ("numeric", ["c_dec52", "c_f32", "c_f64", "c_i32", "c_u64"]),
        ("struct", ["c_struct"]),
        ("temporal", ["c_timestamp", "c_date"]),
    ],
)
def test_of_type_abstract(name, expected):
    t = ibis.table(
        dict(
            c_array="array<int>",
            c_dec52="decimal(5, 2)",
            c_f32="float32",
            c_f64="float64",
            c_point="point",
            c_i32="int32",
            c_u64="uint64",
            c_map="map<string,int>",
            c_struct="struct<a:int>",
            c_timestamp="timestamp",
            c_date="date",
        )
    )
    assert t.select(s.of_type(name)).equals(t.select(*expected))


@pytest.mark.parametrize(
    ("prefixes", "expected"),
    [("a", ("a",)), (("a", "e"), ("a", "e"))],
    ids=["string", "tuple"],
)
def test_startswith(t, prefixes, expected):
    assert t.select(s.startswith(prefixes)).equals(t.select(*expected))


def test_endswith(t):
    assert t.select(s.endswith(("a", "d"))).equals(t.select("a", "d", "ga"))


def test_contains(t):
    assert t.select(s.contains("a")).equals(t.select("a", "ga"))


@pytest.mark.parametrize(
    ("rx", "expected"),
    [("e|f", ("e", "f")), (re.compile("e|f"), ("e", "f"))],
    ids=["string", "pattern"],
)
def test_matches(t, rx, expected):
    assert t.select(s.matches(rx)).equals(t.select(expected))


def test_compose_or(t):
    assert t.select(s.contains("a") | s.startswith("d")).equals(
        t.select("a", "d", "ga")
    )


def test_compose_and(t):
    assert t.select(s.contains("a") & s.contains("g")).equals(t.select("ga"))


def test_compose_not(t):
    assert t.select(~s.numeric()).equals(t.select("b", "c", "d", "g", "ga"))


@pytest.fixture
def penguins():
    return ibis.table(
        dict(
            species="string",
            island="string",
            bill_length_mm="float64",
            bill_depth_mm="float64",
            flipper_length_mm="int64",
            body_mass_g="int64",
            sex="string",
            year="int64",
        ),
        name="penguins",
    )


def zscore(c):
    return (c - c.mean()) / c.std()


@pytest.mark.parametrize(
    "expr_func",
    [
        lambda t: t.select(
            s.across(s.numeric() & ~s.c("year"), (_ - _.mean()) / _.std())
        ),
        lambda t: t.select(s.across(s.numeric() & ~s.c("year"), zscore)),
        lambda t: t.select(
            s.across(s.numeric() & ~s.c(t.year), (_ - _.mean()) / _.std())
        ),
        lambda t: t.select(s.across(s.numeric() & ~s.c(t.year), zscore)),
    ],
    ids=["deferred", "func", "deferred-column-ref", "func-column-ref"],
)
def test_across_select(penguins, expr_func):
    expr = expr_func(penguins)
    expected = penguins.select(
        bill_length_mm=zscore(_.bill_length_mm),
        bill_depth_mm=zscore(_.bill_depth_mm),
        flipper_length_mm=zscore(_.flipper_length_mm),
        body_mass_g=zscore(_.body_mass_g),
    )
    assert expr.equals(expected)


@pytest.mark.parametrize(
    "expr_func",
    [
        lambda t: t.mutate(
            s.across(s.numeric() & ~s.c("year"), (_ - _.mean()) / _.std())
        ),
        lambda t: t.mutate(s.across(s.numeric() & ~s.c("year"), zscore)),
    ],
    ids=["deferred", "func"],
)
def test_across_mutate(penguins, expr_func):
    expr = expr_func(penguins)
    expected = penguins.mutate(
        bill_length_mm=zscore(_.bill_length_mm),
        bill_depth_mm=zscore(_.bill_depth_mm),
        flipper_length_mm=zscore(_.flipper_length_mm),
        body_mass_g=zscore(_.body_mass_g),
    )
    assert expr.equals(expected)


@pytest.mark.parametrize(
    "expr_func",
    [
        lambda t: t.agg(s.across(s.numeric() & ~s.c("year"), _.mean())),
        lambda t: t.agg(s.across(s.numeric() & ~s.c("year"), lambda c: c.mean())),
    ],
    ids=["deferred", "func"],
)
def test_across_agg(penguins, expr_func):
    expr = expr_func(penguins)
    expected = penguins.agg(
        bill_length_mm=_.bill_length_mm.mean(),
        bill_depth_mm=_.bill_depth_mm.mean(),
        flipper_length_mm=_.flipper_length_mm.mean(),
        body_mass_g=_.body_mass_g.mean(),
    )
    assert expr.equals(expected)


@pytest.mark.parametrize(
    "expr_func",
    [
        lambda t: t.group_by("species").select(
            s.across(s.numeric() & ~s.c("year"), (_ - _.mean()) / _.std())
        ),
        lambda t: t.group_by("species").select(
            s.across(s.numeric() & ~s.c("year"), zscore)
        ),
    ],
    ids=["deferred", "func"],
)
def test_across_group_by_select(penguins, expr_func):
    expr = expr_func(penguins)
    expected = penguins.group_by("species").select(
        bill_length_mm=zscore(_.bill_length_mm),
        bill_depth_mm=zscore(_.bill_depth_mm),
        flipper_length_mm=zscore(_.flipper_length_mm),
        body_mass_g=zscore(_.body_mass_g),
    )
    assert expr.equals(expected)


@pytest.mark.parametrize(
    "expr_func",
    [
        lambda t: t.group_by("species").mutate(
            s.across(s.numeric() & ~s.c("year"), (_ - _.mean()) / _.std())
        ),
        lambda t: t.group_by("species").mutate(
            s.across(s.numeric() & ~s.c("year"), zscore)
        ),
    ],
    ids=["deferred", "func"],
)
def test_across_group_by_mutate(penguins, expr_func):
    expr = expr_func(penguins)
    expected = penguins.group_by("species").mutate(
        bill_length_mm=zscore(_.bill_length_mm),
        bill_depth_mm=zscore(_.bill_depth_mm),
        flipper_length_mm=zscore(_.flipper_length_mm),
        body_mass_g=zscore(_.body_mass_g),
    )
    assert expr.equals(expected)


@pytest.mark.parametrize(
    "expr_func",
    [
        lambda t: t.group_by("species").agg(
            s.across(s.numeric() & ~s.c("year"), _.mean())
        ),
        lambda t: t.group_by("species").agg(
            s.across(s.numeric() & ~s.c("year"), lambda c: c.mean())
        ),
    ],
    ids=["deferred", "func"],
)
def test_across_group_by_agg(penguins, expr_func):
    expr = expr_func(penguins)
    expected = penguins.group_by("species").agg(
        bill_length_mm=_.bill_length_mm.mean(),
        bill_depth_mm=_.bill_depth_mm.mean(),
        flipper_length_mm=_.flipper_length_mm.mean(),
        body_mass_g=_.body_mass_g.mean(),
    )
    assert expr.equals(expected)


@pytest.mark.parametrize(
    "expr_func",
    [
        lambda t: t.group_by(~s.numeric()).agg(
            s.across(s.numeric() & ~s.c("year"), _.mean())
        ),
        lambda t: t.group_by(~s.numeric()).agg(
            s.across(s.numeric() & ~s.c("year"), lambda c: c.mean())
        ),
    ],
    ids=["deferred", "func"],
)
def test_across_group_by_agg_with_grouped_selectors(penguins, expr_func):
    expr = expr_func(penguins)
    expected = penguins.group_by(["species", "island", "sex"]).agg(
        bill_length_mm=_.bill_length_mm.mean(),
        bill_depth_mm=_.bill_depth_mm.mean(),
        flipper_length_mm=_.flipper_length_mm.mean(),
        body_mass_g=_.body_mass_g.mean(),
    )
    assert expr.equals(expected)


def test_across_list(penguins):
    expr = penguins.agg(s.across(["species", "island"], lambda c: c.count()))
    expected = penguins.agg(species=_.species.count(), island=_.island.count())
    assert expr.equals(expected)


def test_across_str(penguins):
    expr = penguins.agg(s.across("species", lambda c: c.count()))
    expected = penguins.agg(species=_.species.count())
    assert expr.equals(expected)


def test_if_all(penguins):
    expr = penguins.filter(s.if_all(s.numeric() & ~s.c("year"), _ > 5))
    expected = penguins.filter(
        (_.bill_length_mm > 5)
        & (_.bill_depth_mm > 5)
        & (_.flipper_length_mm > 5)
        & (_.body_mass_g > 5)
    )
    assert expr.equals(expected)


def test_if_any(penguins):
    expr = penguins.filter(s.if_any(s.numeric() & ~s.c("year"), _ > 5))
    expected = penguins.filter(
        (_.bill_length_mm > 5)
        | (_.bill_depth_mm > 5)
        | (_.flipper_length_mm > 5)
        | (_.body_mass_g > 5)
    )
    assert expr.equals(expected)


def test_negate_range(penguins):
    assert penguins.select(~s.r[3:]).equals(penguins.select(0, 1, 2))


def test_string_range_start(penguins):
    assert penguins.select(s.r["island":5]).equals(
        penguins.select(penguins.columns[penguins.columns.index("island") : 5])
    )


def test_string_range_end(penguins):
    assert penguins.select(s.r[:"island"]).equals(
        penguins.select(penguins.columns[: penguins.columns.index("island") + 1])
    )


def test_string_element(penguins):
    assert penguins.select(~s.r["island"]).equals(
        penguins.select([c for c in penguins.columns if c != "island"])
    )


def test_first(penguins):
    assert penguins.select(s.first()).equals(penguins.select(penguins.columns[0]))


def test_last(penguins):
    assert penguins.select(s.last()).equals(penguins.select(penguins.columns[-1]))


def test_all(penguins):
    assert penguins.select(s.all()).equals(penguins.select(penguins.columns))


@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        param([0, 1, 2], (0, 1, 2), id="int_tuple"),
        param(~s.r[[3, 4, 5]], sorted(set(range(8)) - {3, 4, 5}), id="neg_int_list"),
        param(~s.r[3, 4, 5], sorted(set(range(8)) - {3, 4, 5}), id="neg_int_tuple"),
        param(s.r["island", "year"], ("island", "year"), id="string_tuple"),
        param(s.r[["island", "year"]], ("island", "year"), id="string_list"),
        param(iter(["island", 4, "year"]), ("island", 4, "year"), id="mixed_iterable"),
    ],
)
def test_sequence(penguins, seq, expected):
    assert penguins.select(seq).equals(penguins.select(*expected))


def test_names_callable(penguins):
    expr = penguins.select(
        s.across(
            s.numeric() & ~s.c("year"),
            func=dict(cast=_.cast("float32")),
            names=lambda col, fn: f"{fn}({col})",
        )
    )
    expected = penguins.select(
        **{
            "cast(bill_length_mm)": _.bill_length_mm.cast("float32"),
            "cast(bill_depth_mm)": _.bill_depth_mm.cast("float32"),
            "cast(flipper_length_mm)": _.flipper_length_mm.cast("float32"),
            "cast(body_mass_g)": _.body_mass_g.cast("float32"),
        }
    )
    assert expr.equals(expected)


def test_names_format_string(penguins):
    expr = penguins.select(
        s.across(
            s.numeric() & ~s.c("year"),
            func=dict(cast=_.cast("float32")),
            names="{fn}({col})",
        )
    )
    expected = penguins.select(
        **{
            "cast(bill_length_mm)": _.bill_length_mm.cast("float32"),
            "cast(bill_depth_mm)": _.bill_depth_mm.cast("float32"),
            "cast(flipper_length_mm)": _.flipper_length_mm.cast("float32"),
            "cast(body_mass_g)": _.body_mass_g.cast("float32"),
        }
    )
    assert expr.equals(expected)


def test_all_of(penguins):
    expr = penguins.select(s.all_of(s.numeric(), ~s.c("year")))
    expected = penguins.select(
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"
    )
    assert expr.equals(expected)


def test_all_of_string_list(penguins):
    # a bit silly, but robust nonetheless
    expr = penguins.select(s.all_of("year", "year"))
    expected = penguins.select("year")
    assert expr.equals(expected)


def test_any_of(penguins):
    expr = penguins.select(s.any_of(s.startswith("bill"), s.c("year")))
    expected = penguins.select("bill_length_mm", "bill_depth_mm", "year")
    assert expr.equals(expected)


def test_any_of_string_list(penguins):
    expr = penguins.select(s.any_of("year", "body_mass_g", s.matches("length")))
    expected = penguins.select(
        "bill_length_mm", "flipper_length_mm", "body_mass_g", "year"
    )
    assert expr.equals(expected)


def test_c_error_on_misspelled_column(penguins):
    match = "Columns .+ are not present"

    sel = s.c("inland")
    with pytest.raises(exc.IbisInputError, match=match):
        penguins.select(sel)

    sel = s.any_of(s.c("inland"), s.c("island"))
    with pytest.raises(exc.IbisInputError, match=match):
        penguins.select(sel)

    sel = s.any_of(s.c("island"), s.c("inland"))
    with pytest.raises(exc.IbisInputError, match=match):
        penguins.select(sel)

    sel = s.any_of(s.c("island", "inland"))
    with pytest.raises(exc.IbisInputError, match=match):
        penguins.select(sel)


def test_order_by_with_selectors(penguins):
    expr = penguins.order_by(s.of_type("string"))
    assert tuple(key.name for key in expr.op().sort_keys) == (
        "species",
        "island",
        "sex",
    )

    expr = penguins.order_by(s.all())
    assert tuple(key.name for key in expr.op().sort_keys) == tuple(expr.columns)

    with pytest.raises(exc.IbisError):
        penguins.order_by(~s.all())
