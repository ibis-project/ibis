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
call_outcome = ibis.table(
    name="call_outcome", schema={"outcome_text": "string", "id": "int64"}
)
employee = ibis.table(
    name="employee",
    schema={"first_name": "string", "last_name": "string", "id": "int64"},
)
innerjoin = employee.inner_join(call, employee.id == call.employee_id)

result = (
    innerjoin.inner_join(call_outcome, call.call_outcome_id == call_outcome.id)
    .select(
        [
            innerjoin.first_name,
            innerjoin.last_name,
            innerjoin.id,
            innerjoin.start_time,
            innerjoin.end_time,
            innerjoin.employee_id,
            innerjoin.call_outcome_id,
            innerjoin.call_attempts,
            call_outcome.outcome_text,
            call_outcome.id.name("id_right"),
        ]
    )
    .group_by(call.employee_id)
    .aggregate(call.call_attempts.mean().name("avg_attempts"))
)
