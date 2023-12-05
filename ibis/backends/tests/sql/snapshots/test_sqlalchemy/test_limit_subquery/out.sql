SELECT
  t0.c,
  t0.f,
  t0.foo_id,
  t0.bar_id
FROM (
  SELECT
    t1.c AS c,
    t1.f AS f,
    t1.foo_id AS foo_id,
    t1.bar_id AS bar_id
  FROM star1 AS t1
  LIMIT 10
) AS t0
WHERE
  t0.f > 0