SELECT
  t3.foo_id AS foo_id,
  SUM(t3.value1) AS total
FROM (
  SELECT
    t0.c AS c,
    t0.f AS f,
    t0.foo_id AS foo_id,
    t0.bar_id AS bar_id,
    t1.value1 AS value1
  FROM star1 AS t0
  INNER JOIN star2 AS t1
    ON t0.foo_id = t1.foo_id
) AS t3
GROUP BY
  1