import ibis


star1 = ibis.table(
    name="star1",
    schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
)
star2 = ibis.table(
    name="star2", schema={"foo_id": "string", "value1": "float64", "value3": "float64"}
)
agg = star1.aggregate([star1.f.sum().name("total")], by=[star1.foo_id])
joinchain = agg.inner_join(star2, agg.foo_id == star2.foo_id).select(
    agg.foo_id, agg.total, star2.value1
)
f = joinchain.filter(joinchain.total > 100)

result = f.order_by(f.total.desc())
