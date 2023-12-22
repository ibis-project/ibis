SELECT
  t5.c,
  t5.f,
  t5.foo_id,
  t5.bar_id,
  t5.value1,
  t5.value3
FROM (
  SELECT
    t2.c,
    t2.f,
    t2.foo_id,
    t2.bar_id,
    t3.value1,
    t3.value3
  FROM star1 AS t2
  INNER JOIN star2 AS t3
    ON t2.foo_id = t3.foo_id
) AS t5
WHERE
  t5.f > CAST(0 AS TINYINT) AND t5.value3 < CAST(1000 AS SMALLINT)