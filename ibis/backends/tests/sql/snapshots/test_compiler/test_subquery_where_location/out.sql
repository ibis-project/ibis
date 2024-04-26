SELECT
  COUNT("t3"."foo") AS "Count(foo)"
FROM (
  SELECT
    "t2"."string_col",
    SUM("t2"."float_col") AS "foo"
  FROM (
    SELECT
      "t1"."float_col",
      "t1"."timestamp_col",
      "t1"."int_col",
      "t1"."string_col"
    FROM (
      SELECT
        "t0"."float_col",
        "t0"."timestamp_col",
        "t0"."int_col",
        "t0"."string_col"
      FROM "alltypes" AS "t0"
    ) AS "t1"
    WHERE
      "t1"."timestamp_col" < MAKE_TIMESTAMP(2014, 1, 1, 0, 0, 0.0)
  ) AS "t2"
  GROUP BY
    1
) AS "t3"