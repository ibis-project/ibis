import ibis


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
employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)
proj = employee.inner_join(call, employee.id == call.employee_id).filter(
    employee.id < 5
)

result = proj.select(
    [
        proj.first_name,
        proj.last_name,
        proj.id,
        call.start_time,
        call.end_time,
        call.employee_id,
        call.call_outcome_id,
        call.call_attempts,
        proj.first_name.name("first"),
    ]
).order_by(proj.id.desc())
