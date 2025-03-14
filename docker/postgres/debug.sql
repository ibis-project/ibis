SELECT
  "pid",
  "application_name",
  "backend_start",
  "state",
  "wait_event",
  "wait_event_type",
  "query"
FROM "pg_stat_activity"
WHERE
  "backend_type" = 'client backend' AND "state" = 'idle in transaction'
ORDER BY
  "backend_start"
