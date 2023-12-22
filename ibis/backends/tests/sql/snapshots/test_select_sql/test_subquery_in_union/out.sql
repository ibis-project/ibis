SELECT
  t8.a,
  t8.g,
  t8.metric
FROM (
  SELECT
    t2.a,
    t2.g,
    t2.metric
  FROM (
    SELECT
      t0.a,
      t0.g,
      SUM(t0.f) AS metric
    FROM alltypes AS t0
    GROUP BY
      1,
      2
  ) AS t2
  INNER JOIN (
    SELECT
      t0.a,
      t0.g,
      SUM(t0.f) AS metric
    FROM alltypes AS t0
    GROUP BY
      1,
      2
  ) AS t4
    ON t2.g = t4.g
  UNION ALL
  SELECT
    t2.a,
    t2.g,
    t2.metric
  FROM (
    SELECT
      t0.a,
      t0.g,
      SUM(t0.f) AS metric
    FROM alltypes AS t0
    GROUP BY
      1,
      2
  ) AS t2
  INNER JOIN (
    SELECT
      t0.a,
      t0.g,
      SUM(t0.f) AS metric
    FROM alltypes AS t0
    GROUP BY
      1,
      2
  ) AS t4
    ON t2.g = t4.g
) AS t8