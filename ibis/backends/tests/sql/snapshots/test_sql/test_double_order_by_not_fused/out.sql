SELECT
  "t1"."a",
  "t1"."b"
FROM (
  SELECT
    "t0"."a",
    "t0"."b"
  FROM "t" AS "t0"
  ORDER BY
    "t0"."a" ASC
) AS "t1"
ORDER BY
  "t1"."b" DESC