SELECT
  "t2"."string_col"
FROM (
  SELECT
    "t1"."string_col",
    SUM("t1"."float_col") AS "total"
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
      "t0"."int_col" > 0
  ) AS "t1"
  GROUP BY
    "t1"."string_col"
) AS "t2"