SELECT
  "t5"."key",
  "t5"."value"
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t1"."string_col" AS "key",
      CAST("t1"."float_col" AS DOUBLE) AS "value"
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
        "t0"."int_col" > CAST(0 AS TINYINT)
    ) AS "t1"
  ) AS "t4"
  UNION
  SELECT
    *
  FROM (
    SELECT
      "t2"."string_col" AS "key",
      "t2"."double_col" AS "value"
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
        "t0"."int_col" <= CAST(0 AS TINYINT)
    ) AS "t2"
  ) AS "t3"
) AS "t5"