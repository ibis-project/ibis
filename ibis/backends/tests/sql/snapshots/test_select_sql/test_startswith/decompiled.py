import ibis


result = (
    ibis.table(
        name="star1",
        schema={"c": "int32", "f": "float64", "foo_id": "string", "bar_id": "string"},
    )
    .foo_id.startswith(ibis.literal("foo"))
    .name("tmp")
)
