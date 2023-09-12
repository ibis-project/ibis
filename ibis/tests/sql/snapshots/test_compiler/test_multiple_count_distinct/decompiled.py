import ibis


functional_alltypes = ibis.table(
    name="functional_alltypes",
    schema={
        "id": "int32",
        "bool_col": "boolean",
        "tinyint_col": "int8",
        "smallint_col": "int16",
        "int_col": "int32",
        "bigint_col": "int64",
        "float_col": "float32",
        "double_col": "float64",
        "date_string_col": "string",
        "string_col": "string",
        "timestamp_col": "timestamp",
        "year": "int32",
        "month": "int32",
    },
)

result = functional_alltypes.group_by(functional_alltypes.string_col).aggregate(
    [
        functional_alltypes.int_col.nunique().name("int_card"),
        functional_alltypes.smallint_col.nunique().name("smallint_card"),
    ]
)
