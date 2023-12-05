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
proj = alltypes.select(
    [alltypes.float_col, alltypes.timestamp_col, alltypes.int_col, alltypes.string_col]
).filter(alltypes.timestamp_col < param.name("my_param"))
agg = proj.group_by(proj.string_col).aggregate(proj.float_col.sum().name("foo"))

result = agg.foo.count()
