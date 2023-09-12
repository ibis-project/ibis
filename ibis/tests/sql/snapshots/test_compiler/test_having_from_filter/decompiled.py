import ibis


t = ibis.table(name="t", schema={"a": "int64", "b": "string"})

result = (
    t.filter(t.b == "m")
    .group_by(t.b)
    .having(t.a.max() == 2)
    .aggregate(t.a.sum().name("sum"))
)
