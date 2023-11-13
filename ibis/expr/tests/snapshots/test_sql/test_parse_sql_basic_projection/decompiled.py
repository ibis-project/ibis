import ibis


employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)
proj = employee.filter(employee.id < 5)

result = proj.select(
    [proj.first_name, proj.last_name, proj.id, proj.first_name.name("first")]
).order_by(proj.id.desc())
