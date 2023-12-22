import ibis


alltypes = ibis.table(
    name="alltypes",
    schema={
        "a": "int8",
        "b": "int16",
        "c": "int32",
        "d": "int64",
        "e": "float32",
        "f": "float64",
        "g": "string",
        "h": "boolean",
        "i": "timestamp",
        "j": "date",
        "k": "time",
    },
)
agg = alltypes.aggregate([alltypes.f.sum().name("metric")], by=[alltypes.a, alltypes.g])
selfreference = agg.view()
joinchain = agg.inner_join(selfreference, agg.g == selfreference.g).select(
    agg.a, agg.g, agg.metric
)
selfreference1 = joinchain.view()

result = joinchain.union(selfreference1)
