import ibis


tbl = ibis.table(
    name="tbl", schema={"foo": "int32", "bar": "int64", "value": "float64"}
)
f = tbl.filter(tbl.value > 0)

result = f.select(
    f.foo, f.bar, f.value, (f.foo + f.bar).name("baz"), (f.foo * 2).name("qux")
)
