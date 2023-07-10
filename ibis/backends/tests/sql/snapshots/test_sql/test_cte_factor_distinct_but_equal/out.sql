SELECT
  t2.g AS g,
  t2.metric AS metric
FROM (
  SELECT
    t0.g AS g,
    SUM(t0.f) AS metric
  FROM alltypes AS t0
  GROUP BY
    1
) AS t2
INNER JOIN (
  SELECT
    t1.g AS g,
    SUM(t1.f) AS metric
  FROM alltypes AS t1
  GROUP BY
    1
) AS t4
  ON t2.g = t4.g