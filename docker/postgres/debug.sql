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
  "wait_event_type" = 'Client'
ORDER BY
  "backend_start"
