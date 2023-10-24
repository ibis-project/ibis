import ibis


s = ibis.table(name="s", schema={"b": "string"})
t = ibis.table(name="t", schema={"a": "int64", "b": "string", "c": "timestamp"})
lit = ibis.timestamp("2018-01-01 00:00:00")
proj = t.select([t.a, t.b, t.c.name("C")])
proj1 = proj.select([proj.a, proj.b, lit.name("the_date")]).filter(proj.C == lit)
proj2 = proj1.inner_join(s, proj1.b == s.b).select(proj1.a)

result = proj2.filter(proj2.a < 1.0)
