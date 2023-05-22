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
selfreference = functional_alltypes.view()

result = (
    functional_alltypes.inner_join(
        selfreference,
        functional_alltypes.tinyint_col < selfreference.timestamp_col.minute(),
    )
    .select(
        [
            functional_alltypes.id,
            functional_alltypes.bool_col,
            functional_alltypes.tinyint_col,
            functional_alltypes.smallint_col,
            functional_alltypes.int_col,
            functional_alltypes.bigint_col,
            functional_alltypes.float_col,
            functional_alltypes.double_col,
            functional_alltypes.date_string_col,
            functional_alltypes.string_col,
            functional_alltypes.timestamp_col,
            functional_alltypes.year,
            functional_alltypes.month,
            selfreference.id.name("id_right"),
            selfreference.bool_col.name("bool_col_right"),
            selfreference.tinyint_col.name("tinyint_col_right"),
            selfreference.smallint_col.name("smallint_col_right"),
            selfreference.int_col.name("int_col_right"),
            selfreference.bigint_col.name("bigint_col_right"),
            selfreference.float_col.name("float_col_right"),
            selfreference.double_col.name("double_col_right"),
            selfreference.date_string_col.name("date_string_col_right"),
            selfreference.string_col.name("string_col_right"),
            selfreference.timestamp_col.name("timestamp_col_right"),
            selfreference.year.name("year_right"),
            selfreference.month.name("month_right"),
        ]
    )
    .count()
)
