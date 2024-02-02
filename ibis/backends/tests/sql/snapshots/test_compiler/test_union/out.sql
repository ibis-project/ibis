SELECT
  "t3"."key",
  "t3"."value"
FROM (
  SELECT
    "t0"."string_col" AS "key",
    CAST("t0"."float_col" AS DOUBLE) AS "value"
  FROM "functional_alltypes" AS "t0"
  WHERE
    "t0"."int_col" > CAST(0 AS TINYINT)
  UNION
  SELECT
    "t0"."string_col" AS "key",
    "t0"."double_col" AS "value"
  FROM "functional_alltypes" AS "t0"
  WHERE
    "t0"."int_col" <= CAST(0 AS TINYINT)
) AS "t3"