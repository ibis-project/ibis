import ibis


star1 = ibis.table(
    name="star1",
    schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
)
star2 = ibis.table(
    name="star2", schema={"foo_id": "string", "value1": "float64", "value3": "float64"}
)
proj = star1.inner_join(star2, star1.foo_id == star2.foo_id).select(
    [star1, (star1.f - star2.value1).name("diff")]
)

result = proj.filter(proj.diff > 1)
