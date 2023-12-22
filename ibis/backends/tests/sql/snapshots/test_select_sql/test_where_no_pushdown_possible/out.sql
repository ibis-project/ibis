SELECT
  t5.c,
  t5.f,
  t5.foo_id,
  t5.bar_id,
  t5.diff
FROM (
  SELECT
    t2.c,
    t2.f,
    t2.foo_id,
    t2.bar_id,
    t2.f - t3.value1 AS diff
  FROM star1 AS t2
  INNER JOIN star2 AS t3
    ON t2.foo_id = t3.foo_id
) AS t5
WHERE
  t5.diff > CAST(1 AS TINYINT)