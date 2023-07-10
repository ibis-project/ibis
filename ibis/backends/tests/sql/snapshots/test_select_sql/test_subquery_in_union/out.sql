SELECT
  t1.a AS a,
  t1.g AS g,
  t1.metric AS metric
FROM (
  SELECT
    t0.a AS a,
    t0.g AS g,
    SUM(t0.f) AS metric
  FROM alltypes AS t0
  GROUP BY
    1,
    2
) AS t1
INNER JOIN (
  SELECT
    t0.a AS a,
    t0.g AS g,
    SUM(t0.f) AS metric
  FROM alltypes AS t0
  GROUP BY
    1,
    2
) AS t2
  ON t1.g = t2.g
UNION ALL
SELECT
  t1.a AS a,
  t1.g AS g,
  t1.metric AS metric
FROM (
  SELECT
    t0.a AS a,
    t0.g AS g,
    SUM(t0.f) AS metric
  FROM alltypes AS t0
  GROUP BY
    1,
    2
) AS t1
INNER JOIN (
  SELECT
    t0.a AS a,
    t0.g AS g,
    SUM(t0.f) AS metric
  FROM alltypes AS t0
  GROUP BY
    1,
    2
) AS t2
  ON t1.g = t2.g