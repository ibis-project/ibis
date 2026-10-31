SELECT
  "t1"."x",
  "t1"."y",
  "t1"."z",
  CASE WHEN "t1"."y" = "t1"."z" THEN 'big' ELSE 'small' END AS "size"
FROM (
  SELECT
    "t0"."x",
    RANDOM() AS "y",
    RANDOM() AS "z"
  FROM "t" AS "t0"
) AS "t1"