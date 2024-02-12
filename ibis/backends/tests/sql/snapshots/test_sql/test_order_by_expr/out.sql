SELECT
  "t0"."a",
  "t0"."b"
FROM "t" AS "t0"
WHERE
  "t0"."a" = CAST(1 AS TINYINT)
ORDER BY
  "t0"."b" || 'a' ASC