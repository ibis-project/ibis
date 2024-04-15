SELECT
  "t1"."x",
  "t1"."y",
  "t1"."z",
  CASE WHEN "t1"."y" = "t1"."z" THEN 'big' ELSE 'small' END AS "size"
FROM (
  SELECT
    "t0"."x",
    UUID() AS "y",
    UUID() AS "z"
  FROM "t" AS "t0"
) AS "t1"