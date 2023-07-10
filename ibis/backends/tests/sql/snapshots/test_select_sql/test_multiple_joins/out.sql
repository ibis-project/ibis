SELECT
  t0.c AS c,
  t0.f AS f,
  t0.foo_id AS foo_id,
  t0.bar_id AS bar_id,
  t1.value1 AS value1,
  t2.value2 AS value2
FROM star1 AS t0
LEFT OUTER JOIN star2 AS t1
  ON t0.foo_id = t1.foo_id
INNER JOIN star3 AS t2
  ON t0.bar_id = t2.bar_id