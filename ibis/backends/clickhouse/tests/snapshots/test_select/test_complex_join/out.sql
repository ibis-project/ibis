SELECT
  t0.a AS a,
  t0.b AS b,
  t2.c AS c,
  t2.d AS d,
  t2.c / (
    t0.a - t0.b
  ) AS e
FROM s AS t0
INNER JOIN t AS t2
  ON t0.a = t2.c