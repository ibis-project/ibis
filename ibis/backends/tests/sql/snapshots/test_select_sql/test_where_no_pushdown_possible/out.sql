SELECT
  *
FROM (
  SELECT
    t0.c AS c,
    t0.f AS f,
    t0.foo_id AS foo_id,
    t0.bar_id AS bar_id,
    t0.f - t1.value1 AS diff
  FROM star1 AS t0
  INNER JOIN star2 AS t1
    ON t0.foo_id = t1.foo_id
) AS t3
WHERE
  (
    t3.diff > CAST(1 AS TINYINT)
  )