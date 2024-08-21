SELECT
  "t0"."uuid" AS "uuid",
  minIf("t0"."ts", "t0"."search_level" = 1) AS "min_date"
FROM "t" AS "t0"
GROUP BY
  "t0"."uuid"