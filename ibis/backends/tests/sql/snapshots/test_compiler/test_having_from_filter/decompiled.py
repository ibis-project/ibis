import ibis


t = ibis.table(name="t", schema={"a": "int64", "b": "string"})
f = t.filter(t.b == "m")
agg = f.aggregate([f.a.sum().name("sum"), f.a.max()], by=[f.b])
f1 = agg.filter(agg["Max(a)"] == 2)

result = f1.select(f1.b, f1.sum)
