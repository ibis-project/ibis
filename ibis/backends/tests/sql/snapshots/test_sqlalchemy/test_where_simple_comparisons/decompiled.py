import ibis


star1 = ibis.table(
    name="star1",
    schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
)

result = star1.filter([star1.f > 0, star1.c < (star1.f * 2)])
