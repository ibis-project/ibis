import ibis


alltypes = ibis.table(
    name="alltypes",
    schema={
        "float_col": "float32",
        "timestamp_col": "timestamp",
        "int_col": "int32",
        "string_col": "string",
    },
)
param = ibis.param("timestamp")
f = alltypes.filter(alltypes.timestamp_col < param.name("my_param"))
agg = f.aggregate([f.float_col.sum().name("foo")], by=[f.string_col])

result = agg.foo.count()
