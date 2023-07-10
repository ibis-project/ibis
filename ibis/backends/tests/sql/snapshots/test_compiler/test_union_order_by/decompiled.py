import ibis


t = ibis.table(name="t", schema={"a": "int64", "b": "string"})
s = t.order_by(t.b.asc())

result = s.union(s)
