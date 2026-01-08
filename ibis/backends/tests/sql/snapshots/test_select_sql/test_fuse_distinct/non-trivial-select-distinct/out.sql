SELECT DISTINCT
  "t0"."a",
  "t0"."b" % CAST(2 AS TINYINT) AS "d"
FROM "test" AS "t0"
WHERE
  "t0"."c" > 10