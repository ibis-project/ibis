SELECT
  t0.x1 AS x1,
  t0.y1 AS y1,
  t1.x2 AS x2,
  t6.x3 AS x3,
  t6.y2 AS y2,
  t6.x4 AS x4
FROM t1 AS t0
INNER JOIN t2 AS t1
  ON t0.x1 = t1.x2
INNER JOIN (
  SELECT
    t2.x3 AS x3,
    t2.y2 AS y2,
    t3.x4 AS x4
  FROM t3 AS t2
  INNER JOIN t4 AS t3
    ON t2.x3 = t3.x4
) AS t6
  ON t0.y1 = t6.y2