SELECT
  "t2"."string_col",
  "t2"."CountStar(functional_alltypes)"
FROM (
  SELECT
    "t1"."string_col",
    "t1"."CountStar(functional_alltypes)",
    "t1"."Max(double_col)"
  FROM (
    SELECT
      "t0"."string_col",
      COUNT(*) AS "CountStar(functional_alltypes)",
      MAX("t0"."double_col") AS "Max(double_col)"
    FROM "functional_alltypes" AS "t0"
    GROUP BY
      1
  ) AS "t1"
  WHERE
    "t1"."Max(double_col)" = CAST(1 AS TINYINT)
) AS "t2"