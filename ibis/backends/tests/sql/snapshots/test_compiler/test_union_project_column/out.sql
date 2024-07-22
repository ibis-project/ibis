SELECT
  "t3"."key"
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t0"."string_col" AS "key",
      CAST("t0"."float_col" AS DOUBLE) AS "value"
    FROM "functional_alltypes" AS "t0"
    WHERE
      "t0"."int_col" > 0
  ) AS "t1"
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      "t0"."string_col" AS "key",
      "t0"."double_col" AS "value"
    FROM "functional_alltypes" AS "t0"
    WHERE
      "t0"."int_col" <= 0
  ) AS "t2"
) AS "t3"