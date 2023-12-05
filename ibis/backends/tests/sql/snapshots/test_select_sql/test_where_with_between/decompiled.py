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
lit = ibis.literal(0)

result = alltypes.filter([alltypes.a > lit, alltypes.f.between(lit, ibis.literal(1))])
