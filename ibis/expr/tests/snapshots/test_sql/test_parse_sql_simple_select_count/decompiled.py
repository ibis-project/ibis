import ibis


employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)

result = employee.aggregate([employee.count().name("_col_0")])
