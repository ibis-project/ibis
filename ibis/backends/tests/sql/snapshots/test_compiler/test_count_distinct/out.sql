SELECT
  "t1"."string_col",
  COUNT(DISTINCT "t1"."int_col") AS "nunique"
FROM (
  SELECT
    "t0"."id",
    "t0"."bool_col",
    "t0"."tinyint_col",
    "t0"."smallint_col",
    "t0"."int_col",
    "t0"."bigint_col",
    "t0"."float_col",
    "t0"."double_col",
    "t0"."date_string_col",
    "t0"."string_col",
    "t0"."timestamp_col",
    "t0"."year",
    "t0"."month"
  FROM "functional_alltypes" AS "t0"
  WHERE
    "t0"."bigint_col" > CAST(0 AS TINYINT)
) AS "t1"
GROUP BY
  1