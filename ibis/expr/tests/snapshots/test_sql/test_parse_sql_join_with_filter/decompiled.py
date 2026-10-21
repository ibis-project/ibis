import ibis

employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)
call = ibis.table(
    name="call",
    schema={
        "start_time": "timestamp",
        "end_time": "timestamp",
        "employee_id": "int64",
        "call_outcome_id": "int64",
        "call_attempts": "int64",
    },
)
joinchain = employee.left_join(call, (employee.id == call.employee_id)).select(
    employee.first_name,
    employee.last_name,
    employee.id,
    call.start_time,
    call.end_time,
    call.employee_id,
    call.call_outcome_id,
    call.call_attempts,
)
f = joinchain.filter((joinchain.id < 5))
p = f.select(
    f.first_name,
    f.last_name,
    f.id,
    f.start_time,
    f.end_time,
    f.employee_id,
    f.call_outcome_id,
    f.call_attempts,
    f.first_name.name("first"),
)

result = p.order_by(p.id.desc())
