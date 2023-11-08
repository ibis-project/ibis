import ibis


employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)
f = employee.filter(employee.id < 5)
s = f.order_by(f.id.desc())

result = s.select(s.first_name, s.last_name, s.id, s.first_name.name("first"))
