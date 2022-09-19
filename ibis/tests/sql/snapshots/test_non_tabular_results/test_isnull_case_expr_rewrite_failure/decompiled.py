import ibis


result = (
    ibis.table(
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
    .g.isnull()
    .ifelse(ibis.literal(1), ibis.literal(0))
    .sum()
    .name("sum")
)
