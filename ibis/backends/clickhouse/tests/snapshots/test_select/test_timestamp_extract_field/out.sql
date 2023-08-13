SELECT
  toYear(t0.timestamp_col) AS year,
  toMonth(t0.timestamp_col) AS month,
  toDayOfMonth(t0.timestamp_col) AS day,
  toHour(t0.timestamp_col) AS hour,
  toMinute(t0.timestamp_col) AS minute,
  toSecond(t0.timestamp_col) AS second
FROM functional_alltypes AS t0