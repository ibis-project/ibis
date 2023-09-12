SELECT
  t0.x1,
  t0.y1,
  t1.x2,
  t2.x3,
  t2.y2,
  t2.x4
FROM t1 AS t0
JOIN t2 AS t1
  ON t0.x1 = t1.x2
JOIN (
  SELECT
    t3.x3 AS x3,
    t3.y2 AS y2,
    t4.x4 AS x4
  FROM t3 AS t3
  JOIN t4 AS t4
    ON t3.x3 = t4.x4
) AS t2
  ON t0.y1 = t2.y2