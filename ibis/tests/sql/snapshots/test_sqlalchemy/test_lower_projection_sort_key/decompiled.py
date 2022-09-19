import ibis


star2 = ibis.table(
    name="star2", schema={"foo_id": "string", "value1": "float64", "value3": "float64"}
)
star1 = ibis.table(
    name="star1",
    schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
)
agg = star1.group_by(star1.foo_id).aggregate(star1.f.sum().name("total"))
proj = agg.inner_join(star2, agg.foo_id == star2.foo_id).select([agg, star2.value1])
proj1 = proj.filter(proj.total > 100)

result = proj1.order_by(proj1.total.desc())
