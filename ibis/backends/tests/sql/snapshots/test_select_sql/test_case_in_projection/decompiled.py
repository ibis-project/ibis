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
lit = ibis.literal("foo")
lit1 = ibis.literal("baz")
lit2 = ibis.literal("bar")

result = alltypes.select(
    [
        alltypes.g.case()
        .when(lit, lit2)
        .when(lit1, ibis.literal("qux"))
        .else_(ibis.literal("default"))
        .end()
        .name("col1"),
        ibis.case()
        .when(alltypes.g == lit, lit2)
        .when(alltypes.g == lit1, alltypes.g)
        .else_(ibis.literal(None).cast("string"))
        .end()
        .name("col2"),
        alltypes,
    ]
)
