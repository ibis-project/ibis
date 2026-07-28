SELECT
  "t1"."x" AS "x",
  "t1"."y" AS "y",
  "t1"."z" AS "z",
  CASE WHEN "t1"."y" = "t1"."z" THEN 'big' ELSE 'small' END AS "size"
FROM (
  SELECT
    "t0"."x" AS "x",
    generateUUIDv4() AS "y",
    generateUUIDv4() AS "z"
  FROM "t" AS "t0"
) AS "t1"