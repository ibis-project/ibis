SELECT
  "t4"."a",
  "t4"."b",
  "t4"."c",
  "t4"."d",
  "t4"."c" / (
    "t4"."a" - "t4"."b"
  ) AS "e"
FROM (
  SELECT
    "t2"."a",
    "t2"."b",
    "t3"."c",
    "t3"."d"
  FROM "s" AS "t2"
  INNER JOIN "t" AS "t3"
    ON "t2"."a" = "t3"."c"
) AS "t4"