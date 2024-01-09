SELECT
  t6.string_col,
  t6.metric
FROM (
  SELECT
    t4.string_col,
    t4.metric
  FROM (
    SELECT
      t0.string_col,
      SUM(t0.double_col) AS metric
    FROM functional_alltypes AS t0
    GROUP BY
      1
    UNION DISTINCT
    SELECT
      t0.string_col,
      SUM(t0.double_col) AS metric
    FROM functional_alltypes AS t0
    GROUP BY
      1
  ) AS t4
  UNION DISTINCT
  SELECT
    t0.string_col,
    SUM(t0.double_col) AS metric
  FROM functional_alltypes AS t0
  GROUP BY
    1
) AS t6