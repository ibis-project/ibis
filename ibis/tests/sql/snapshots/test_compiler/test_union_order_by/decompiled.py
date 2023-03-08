import ibis


t = ibis.table(name="t", schema={"a": "int64", "b": "string"})
proj = t.order_by(t.b.asc())

result = proj.union(proj)
