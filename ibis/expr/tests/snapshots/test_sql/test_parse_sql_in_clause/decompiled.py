import ibis


employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)

result = employee.select(employee.first_name).filter(
    employee.first_name.isin(
        (
            ibis.literal("Graham"),
            ibis.literal("John"),
            ibis.literal("Terry"),
            ibis.literal("Eric"),
            ibis.literal("Michael"),
        )
    )
)
