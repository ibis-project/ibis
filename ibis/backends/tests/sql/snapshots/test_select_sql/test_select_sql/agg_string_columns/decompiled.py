import ibis


star1 = ibis.table(
    name="star1",
    schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
)

result = star1.aggregate([star1.f.sum().name("total")], by=[star1.foo_id, star1.bar_id])
