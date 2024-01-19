WITH t6 AS (
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
), t1 AS (
  SELECT
    t0.a,
    t0.g,
    SUM(t0.f) AS metric
  FROM alltypes AS t0
  GROUP BY
    1,
    2
)
SELECT
  t9.a,
  t9.g,
  t9.metric
FROM (
  SELECT
    *
  FROM t6 AS t7
  UNION ALL
  SELECT
    *
  FROM t6 AS t8
) AS t9