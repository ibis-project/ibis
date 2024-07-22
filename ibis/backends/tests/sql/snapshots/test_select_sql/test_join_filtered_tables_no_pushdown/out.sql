SELECT
  "t4"."value_a",
  "t5"."value_b"
FROM (
  SELECT
    *
  FROM "a" AS "t0"
  WHERE
    "t0"."year" = 2016 AND "t0"."month" = 2 AND "t0"."day" = 29
) AS "t4"
LEFT OUTER JOIN (
  SELECT
    *
  FROM "b" AS "t1"
  WHERE
    "t1"."year" = 2016 AND "t1"."month" = 2 AND "t1"."day" = 29
) AS "t5"
  ON "t4"."year" = "t5"."year"
  AND "t4"."month" = "t5"."month"
  AND "t4"."day" = "t5"."day"