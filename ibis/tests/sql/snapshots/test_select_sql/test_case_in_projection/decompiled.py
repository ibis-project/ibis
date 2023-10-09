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
lit = ibis.literal("bar")
equals = alltypes.g == "foo"
equals1 = alltypes.g == "baz"

result = alltypes.select(
    [
        ibis.cases(
            (equals, lit), (equals1, ibis.literal("qux")), else_=ibis.literal("default")
        ).name("col1"),
        ibis.cases(
            (equals, lit),
            (equals1, alltypes.g),
            else_=ibis.literal(None).cast("string"),
        ).name("col2"),
        alltypes,
    ]
)
