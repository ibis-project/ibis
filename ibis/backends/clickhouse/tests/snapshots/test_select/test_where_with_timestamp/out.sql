SELECT
  t0.uuid,
  minIf(t0.ts, search_level = 1) AS min_date
FROM t AS t0
GROUP BY
  1