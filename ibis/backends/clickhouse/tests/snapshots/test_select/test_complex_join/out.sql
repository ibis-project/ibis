SELECT
  "t4"."a" AS "a",
  "t4"."b" AS "b",
  "t4"."c" AS "c",
  "t4"."d" AS "d",
  "t4"."c" / (
    "t4"."a" - "t4"."b"
  ) AS "e"
FROM (
  SELECT
    "t2"."a" AS "a",
    "t2"."b" AS "b",
    "t3"."c" AS "c",
    "t3"."d" AS "d"
  FROM "s" AS "t2"
  INNER JOIN "t" AS "t3"
    ON "t2"."a" = "t3"."c"
) AS "t4"