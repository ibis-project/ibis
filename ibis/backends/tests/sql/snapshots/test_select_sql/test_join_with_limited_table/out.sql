SELECT
  t2.c AS c,
  t2.f AS f,
  t2.foo_id AS foo_id,
  t2.bar_id AS bar_id
FROM (
  SELECT
    *
  FROM star1 AS t0
  LIMIT 100
) AS t2
INNER JOIN star2 AS t1
  ON t2.foo_id = t1.foo_id