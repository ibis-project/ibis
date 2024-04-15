SELECT
  "t1"."x",
  "t1"."y",
  "t1"."z",
  IFF("t1"."y" = "t1"."z", 'big', 'small') AS "size"
FROM (
  SELECT
    "t0"."x",
    UUID_STRING() AS "y",
    UUID_STRING() AS "z"
  FROM "t" AS "t0"
) AS "t1"