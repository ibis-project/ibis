SELECT
  "t4"."value_a",
  "t5"."value_b"
FROM (
  SELECT
    *
  FROM "a" AS "t0"
  WHERE
    "t0"."year" = CAST(2016 AS SMALLINT)
    AND "t0"."month" = CAST(2 AS TINYINT)
    AND "t0"."day" = CAST(29 AS TINYINT)
) AS "t4"
LEFT OUTER JOIN (
  SELECT
    *
  FROM "b" AS "t1"
  WHERE
    "t1"."year" = CAST(2016 AS SMALLINT)
    AND "t1"."month" = CAST(2 AS TINYINT)
    AND "t1"."day" = CAST(29 AS TINYINT)
) AS "t5"
  ON "t4"."year" = "t5"."year"
  AND "t4"."month" = "t5"."month"
  AND "t4"."day" = "t5"."day"