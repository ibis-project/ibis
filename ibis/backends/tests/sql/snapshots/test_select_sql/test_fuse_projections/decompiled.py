import ibis


tbl = ibis.table(
    name="tbl", schema={"foo": "int32", "bar": "int64", "value": "float64"}
)
alias = (tbl.foo + tbl.bar).name("baz")
proj = tbl.select([tbl, alias])

result = (
    tbl.select([tbl, alias])
    .filter(tbl.value > 0)
    .select([proj, (proj.foo * 2).name("qux")])
)
