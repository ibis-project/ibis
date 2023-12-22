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
call_outcome = ibis.table(
    name="call_outcome", schema={"outcome_text": "string", "id": "int64"}
)
joinchain = (
    employee.inner_join(call, employee.id == call.employee_id)
    .inner_join(call_outcome, call.call_outcome_id == call_outcome.id)
    .select(
        employee.first_name,
        employee.last_name,
        employee.id,
        call.start_time,
        call.end_time,
        call.employee_id,
        call.call_outcome_id,
        call.call_attempts,
        call_outcome.outcome_text,
        call_outcome.id.name("id_right"),
    )
)

result = joinchain.aggregate(
    [joinchain.call_attempts.mean().name("avg_attempts")], by=[joinchain.employee_id]
)
