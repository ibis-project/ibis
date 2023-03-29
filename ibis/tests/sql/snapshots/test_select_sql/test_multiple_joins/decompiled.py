import ibis


star1 = ibis.table(
    name="star1",
    schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
)
star3 = ibis.table(name="star3", schema={"bar_id": "string", "value2": "float64"})
star2 = ibis.table(
    name="star2", schema={"foo_id": "string", "value1": "float64", "value3": "float64"}
)

result = (
    star1.left_join(star2, star1.foo_id == star2.foo_id)
    .select(
        [
            star1.c,
            star1.f,
            star1.foo_id,
            star1.bar_id,
            star2.foo_id.name("foo_id_right"),
            star2.value1,
            star2.value3,
        ]
    )
    .inner_join(star3, star1.bar_id == star3.bar_id)
    .select([star1, star2.value1, star3.value2])
)
