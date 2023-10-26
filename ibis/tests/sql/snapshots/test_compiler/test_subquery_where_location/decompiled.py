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
agg = (
    alltypes.filter(alltypes.timestamp_col < param.name("my_param"))
    .group_by(alltypes.string_col)
    .aggregate(alltypes.float_col.sum().name("foo"))
)

result = agg.foo.count()
