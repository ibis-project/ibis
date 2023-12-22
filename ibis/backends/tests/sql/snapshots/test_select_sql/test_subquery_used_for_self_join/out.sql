SELECT
  t6.g,
  MAX(t6.total - t6.total_right) AS metric
FROM (
  SELECT
    t2.g,
    t2.a,
    t2.b,
    t2.total,
    t4.g AS g_right,
    t4.a AS a_right,
    t4.b AS b_right,
    t4.total AS total_right
  FROM (
    SELECT
      t0.g,
      t0.a,
      t0.b,
      SUM(t0.f) AS total
    FROM alltypes AS t0
    GROUP BY
      1,
      2,
      3
  ) AS t2
  INNER JOIN (
    SELECT
      t0.g,
      t0.a,
      t0.b,
      SUM(t0.f) AS total
    FROM alltypes AS t0
    GROUP BY
      1,
      2,
      3
  ) AS t4
    ON t2.a = t4.b
) AS t6
GROUP BY
  1