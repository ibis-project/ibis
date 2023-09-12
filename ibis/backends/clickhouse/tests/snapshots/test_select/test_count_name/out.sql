SELECT
  t0.a,
  ifNull(countIf(NOT t0.b), 0) AS A,
  ifNull(countIf(t0.b), 0) AS B
FROM t AS t0
GROUP BY
  t0.a