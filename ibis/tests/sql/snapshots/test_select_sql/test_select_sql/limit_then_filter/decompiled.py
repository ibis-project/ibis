import ibis


limit = ibis.table(
    name="star1",
    schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
).limit(10)

result = limit.filter(limit.f > 0)
