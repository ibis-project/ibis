SELECT
  "t1"."a",
  "t1"."b"
FROM (
  SELECT
    "t0"."a",
    "t0"."b"
  FROM "t" AS "t0"
  WHERE
    "t0"."a" = CAST(1 AS TINYINT)
) AS "t1"
ORDER BY
  "t1"."b" || 'a' ASC