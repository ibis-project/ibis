SELECT
  *
FROM (
  SELECT
    t0.c AS c,
    t0.f AS f,
    t0.foo_id AS foo_id,
    t0.bar_id AS bar_id,
    t1.value1 AS value1,
    t1.value3 AS value3
  FROM star1 AS t0
  INNER JOIN star2 AS t1
    ON t0.foo_id = t1.foo_id
) AS t3
WHERE
  (
    t3.f > CAST(0 AS TINYINT)
  ) AND (
    t3.value3 < CAST(1000 AS SMALLINT)
  )