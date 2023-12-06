SELECT
  t0.a AS a,
  t0.b AS b,
  t1.c AS c,
  t1.d AS d,
  t1.c / (
    t0.a - t0.b
  ) AS e
FROM unbound_table_0 AS t0
INNER JOIN unbound_table_1 AS t1
  ON t0.a = t1.c