SELECT
  t3.a AS a,
  t3.b AS b,
  t3.c AS c,
  t3.d AS d,
  t3.c / (
    t3.a - t3.b
  ) AS e
FROM (
  SELECT
    t0.a AS a,
    t0.b AS b,
    t1.c AS c,
    t1.d AS d
  FROM s AS t0
  INNER JOIN t AS t1
    ON t0.a = t1.c
) AS t3