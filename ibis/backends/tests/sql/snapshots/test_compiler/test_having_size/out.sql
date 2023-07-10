SELECT
  t2.string_col AS string_col,
  t2."CountStar()" AS "CountStar()"
FROM (
  SELECT
    *
  FROM (
    SELECT
      t0.string_col AS string_col,
      COUNT(*) AS "CountStar()",
      MAX(t0.double_col) AS "Max(double_col)"
    FROM functional_alltypes AS t0
    GROUP BY
      1
  ) AS t1
  WHERE
    (
      t1."Max(double_col)" = CAST(1 AS TINYINT)
    )
) AS t2