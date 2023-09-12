import ibis


star1 = ibis.table(
    name="star1",
    schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
)

result = star1.group_by(star1.foo_id).aggregate(star1.f.sum().name("total"))
