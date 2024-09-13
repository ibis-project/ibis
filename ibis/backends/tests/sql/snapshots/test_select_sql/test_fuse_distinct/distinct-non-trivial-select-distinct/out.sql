SELECT DISTINCT
  "t1"."a",
  "t1"."b" % 2 AS "d"
FROM (
  SELECT DISTINCT
    "t0"."a",
    "t0"."b",
    "t0"."c"
  FROM "test" AS "t0"
  WHERE
    "t0"."c" > 10
) AS "t1"