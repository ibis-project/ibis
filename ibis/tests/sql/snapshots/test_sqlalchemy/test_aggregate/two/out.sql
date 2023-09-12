SELECT
  t0.foo_id,
  t0.bar_id,
  SUM(t0.f) AS total
FROM star1 AS t0
GROUP BY
  1,
  2