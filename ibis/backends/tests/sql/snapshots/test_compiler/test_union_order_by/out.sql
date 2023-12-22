SELECT
  t2.a,
  t2.b
FROM (
  SELECT
    t0.a,
    t0.b
  FROM t AS t0
  ORDER BY
    t0.b ASC
  UNION ALL
  SELECT
    t0.a,
    t0.b
  FROM t AS t0
  ORDER BY
    t0.b ASC
) AS t2