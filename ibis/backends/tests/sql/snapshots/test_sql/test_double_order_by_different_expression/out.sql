SELECT
  "t0"."a",
  "t0"."b",
  "t0"."c"
FROM "t" AS "t0"
ORDER BY
  "t0"."b" ASC,
  "t0"."a" + CAST(1 AS TINYINT) DESC,
  "t0"."a" ASC,
  "t0"."c" ASC