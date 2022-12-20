import ibis


s = ibis.table(name="s", schema={"b": "string"})
lit = ibis.timestamp("2018-01-01 00:00:00")
t = ibis.table(name="t", schema={"a": "int64", "b": "string", "c": "timestamp"})
proj = t.select([t.a, t.b, t.c.name("C")])
proj1 = proj.filter(proj.C == lit)
proj2 = proj1.select([proj1.a, proj1.b, lit.name("the_date")])
proj3 = proj2.inner_join(s, proj2.b == s.b).select(proj2.a)

result = proj3.filter(proj3.a < 1.0)
