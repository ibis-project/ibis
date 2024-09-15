SELECT DISTINCT
  "t0"."a",
  "t0"."b" % 2 AS "d"
FROM "test" AS "t0"
WHERE
  "t0"."c" > 10