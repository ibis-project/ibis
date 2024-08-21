SELECT
  "t0"."a" AS "a",
  COALESCE(countIf(NOT (
    "t0"."b"
  )), 0) AS "A",
  COALESCE(countIf("t0"."b"), 0) AS "B"
FROM "t" AS "t0"
GROUP BY
  "t0"."a"