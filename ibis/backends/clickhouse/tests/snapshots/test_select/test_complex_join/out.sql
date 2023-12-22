SELECT
  t5.a,
  t5.b,
  t5.c,
  t5.d,
  t5.c / (
    t5.a - t5.b
  ) AS e
FROM (
  SELECT
    t2.a,
    t2.b,
    t3.c,
    t3.d
  FROM s AS t2
  INNER JOIN t AS t3
    ON t2.a = t3.c
) AS t5