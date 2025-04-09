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

result = (
    employee.inner_join(call, [(employee.id == call.employee_id), ibis.literal(True)])
    .select(
        employee.first_name,
        employee.last_name,
        employee.id,
        call.start_time,
        call.end_time,
        call.employee_id,
        call.call_outcome_id,
        call.call_attempts,
        employee.first_name.name("first"),
    )
    .limit(3)
)
