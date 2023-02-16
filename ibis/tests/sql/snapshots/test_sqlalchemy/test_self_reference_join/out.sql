SELECT
  t0.c,
  t0.f,
  t0.foo_id,
  t0.bar_id
FROM star1 AS t0
JOIN star1 AS t1
  ON t0.foo_id = t1.bar_id