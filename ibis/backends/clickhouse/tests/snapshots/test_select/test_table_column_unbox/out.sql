SELECT
  "t2"."string_col" AS "string_col"
FROM (
  SELECT
    "t1"."string_col" AS "string_col",
    SUM("t1"."float_col") AS "total"
  FROM (
    SELECT
      *
    FROM "functional_alltypes" AS "t0"
    WHERE
      "t0"."int_col" > 0
  ) AS "t1"
  GROUP BY
    "t1"."string_col"
) AS "t2"