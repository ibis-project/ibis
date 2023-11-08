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

result = call.aggregate(
    [call.call_attempts.sum().name("attempts")], by=[call.employee_id]
)
