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
            functional_alltypes.id.name("id_x"),
            functional_alltypes.bool_col.name("bool_col_x"),
            functional_alltypes.tinyint_col.name("tinyint_col_x"),
            functional_alltypes.smallint_col.name("smallint_col_x"),
            functional_alltypes.int_col.name("int_col_x"),
            functional_alltypes.bigint_col.name("bigint_col_x"),
            functional_alltypes.float_col.name("float_col_x"),
            functional_alltypes.double_col.name("double_col_x"),
            functional_alltypes.date_string_col.name("date_string_col_x"),
            functional_alltypes.string_col.name("string_col_x"),
            functional_alltypes.timestamp_col.name("timestamp_col_x"),
            functional_alltypes.year.name("year_x"),
            functional_alltypes.month.name("month_x"),
            selfreference.id.name("id_y"),
            selfreference.bool_col.name("bool_col_y"),
            selfreference.tinyint_col.name("tinyint_col_y"),
            selfreference.smallint_col.name("smallint_col_y"),
            selfreference.int_col.name("int_col_y"),
            selfreference.bigint_col.name("bigint_col_y"),
            selfreference.float_col.name("float_col_y"),
            selfreference.double_col.name("double_col_y"),
            selfreference.date_string_col.name("date_string_col_y"),
            selfreference.string_col.name("string_col_y"),
            selfreference.timestamp_col.name("timestamp_col_y"),
            selfreference.year.name("year_y"),
            selfreference.month.name("month_y"),
        ]
    )
    .count()
    .name("count")
)
