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
    alltypes.g.cases(
        (lit, lit2), (lit1, ibis.literal("qux")), else_=ibis.literal("default")
    ).name("col1"),
    ibis.cases(
        ((alltypes.g == lit), lit2),
        ((alltypes.g == lit1), alltypes.g),
        else_=ibis.literal(None),
    ).name("col2"),
    alltypes.a,
    alltypes.b,
    alltypes.c,
    alltypes.d,
    alltypes.e,
    alltypes.f,
    alltypes.g,
    alltypes.h,
    alltypes.i,
    alltypes.j,
    alltypes.k,
)
