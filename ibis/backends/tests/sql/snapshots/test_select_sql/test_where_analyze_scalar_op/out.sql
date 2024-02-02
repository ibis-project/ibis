SELECT
  COUNT(*) AS "CountStar()"
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
    "t0"."timestamp_col" < (
      MAKE_TIMESTAMP(2010, 1, 1, 0, 0, 0.0) + INTERVAL '3' MONTH
    )
    AND "t0"."timestamp_col" < (
      CAST(CURRENT_TIMESTAMP AS TIMESTAMP) + INTERVAL '10' DAY
    )
) AS "t1"