import ibis

employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)
f = employee.filter((employee.id < 5))
p = f.select(f.first_name, f.last_name, f.id, f.first_name.name("first"))

result = p.order_by(p.id.desc())
