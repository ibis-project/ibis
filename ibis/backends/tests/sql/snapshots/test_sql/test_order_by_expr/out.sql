SELECT
  *
FROM (
  SELECT
    *
  FROM "t" AS "t0"
  WHERE
    "t0"."a" = CAST(1 AS TINYINT)
) AS "t1"
ORDER BY
  "t1"."b" || 'a' ASC