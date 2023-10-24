import ibis


tbl = ibis.table(
    name="tbl", schema={"foo": "int32", "bar": "int64", "value": "float64"}
)

result = tbl.select([tbl, (tbl.foo * 2).name("qux")]).filter(tbl.value > 0)
