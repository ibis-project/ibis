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
leftjoin = employee.left_join(call, employee.id == call.employee_id)

result = leftjoin.group_by(leftjoin.id).aggregate(
    call.call_attempts.sum().name("attempts")
)
