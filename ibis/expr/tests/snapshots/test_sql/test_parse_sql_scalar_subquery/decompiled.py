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
agg = call.aggregate([call.call_attempts.mean().name("mean")])

result = call.inner_join(
    agg, [agg.mean < call.call_attempts, ibis.literal(True)]
).select(
    call.start_time,
    call.end_time,
    call.employee_id,
    call.call_outcome_id,
    call.call_attempts,
    agg.mean,
)
