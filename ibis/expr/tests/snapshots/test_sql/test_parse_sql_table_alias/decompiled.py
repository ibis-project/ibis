import ibis


employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)

result = employee.select([employee.first_name, employee.last_name, employee.id])
