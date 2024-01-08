WITH t0 AS (
  SELECT
    t2.id AS id,
    t2.ts_tz AS tz,
    t2.ts_no_tz AS no_tz
  FROM x AS t2
)
SELECT
  t0.id,
  CAST(t0.tz AS TIMESTAMPTZ) AS tz,
  CAST(t0.no_tz AS TIMESTAMP) AS no_tz,
  t1.id AS id_right
FROM t0
LEFT OUTER JOIN y AS t1
  ON t0.id = t1.id