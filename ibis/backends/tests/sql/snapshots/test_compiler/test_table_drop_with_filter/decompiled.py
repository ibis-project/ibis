import ibis


lit = ibis.timestamp("2018-01-01 00:00:00")
s = ibis.table(name="s", schema={"b": "string"})
t = ibis.table(name="t", schema={"a": "int64", "b": "string", "c": "timestamp"})
f = t.filter((t.c == lit))
dropcolumns = f.select(f.a, f.b, f.c.name("C")).drop("C")
joinchain = (
    dropcolumns.select(dropcolumns.a, dropcolumns.b, lit.name("the_date"))
    .inner_join(
        s,
        (
            dropcolumns.select(dropcolumns.a, dropcolumns.b, lit.name("the_date")).b
            == s.b
        ),
    )
    .select(dropcolumns.select(dropcolumns.a, dropcolumns.b, lit.name("the_date")).a)
)

result = joinchain.filter((joinchain.a < 1.0))
