SELECT
  "t1"."x",
  "t1"."y",
  "t1"."z",
  IFF("t1"."y" = "t1"."z", 'big', 'small') AS "size"
FROM (
  SELECT
    "t0"."x",
    UNIFORM(TO_DOUBLE(0.0), TO_DOUBLE(1.0), RANDOM()) AS "y",
    UNIFORM(TO_DOUBLE(0.0), TO_DOUBLE(1.0), RANDOM()) AS "z"
  FROM "t" AS "t0"
) AS "t1"