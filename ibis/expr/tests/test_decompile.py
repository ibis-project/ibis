from __future__ import annotations

import pytest

import ibis
import ibis.expr.types as ir
from ibis import _
from ibis.expr.decompile import decompile

t = ibis.table([("a", "int64"), ("b", "string")], name="t")
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

nine_ = (two + one) ** 2


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
        (three, 3),
        (nine, nine_),
    ],
    ids=[
        "top_with_highest_population",
        "overall_population_density",
        "population_density_per_country",
        "three",
        "nine",
    ],
)
def test_basic(expr, expected):
    rendered = decompile(expr)

    locals_ = {}
    exec(rendered, {}, locals_)
    restored = locals_["result"]

    if isinstance(expected, ir.Expr):
        assert restored.equals(expected)
    else:
        assert restored == expected
