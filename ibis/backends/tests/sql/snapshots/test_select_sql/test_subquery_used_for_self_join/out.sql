SELECT
  t4.g AS g,
  MAX(t4.total - t4.total_right) AS metric
FROM (
  SELECT
    t1.g AS g,
    t1.a AS a,
    t1.b AS b,
    t1.total AS total,
    t2.g AS g_right,
    t2.a AS a_right,
    t2.b AS b_right,
    t2.total AS total_right
  FROM (
    SELECT
      t0.g AS g,
      t0.a AS a,
      t0.b AS b,
      SUM(t0.f) AS total
    FROM alltypes AS t0
    GROUP BY
      1,
      2,
      3
  ) AS t1
  INNER JOIN (
    SELECT
      t0.g AS g,
      t0.a AS a,
      t0.b AS b,
      SUM(t0.f) AS total
    FROM alltypes AS t0
    GROUP BY
      1,
      2,
      3
  ) AS t2
    ON t1.a = t2.b
) AS t4
GROUP BY
  1