import ibis


employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)
f = employee.filter(
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

result = f.select(f.first_name)
