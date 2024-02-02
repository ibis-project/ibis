SELECT
  "t7"."p_partkey",
  "t7"."ps_supplycost"
FROM (
  SELECT
    "t3"."p_partkey",
    "t4"."ps_supplycost"
  FROM "part" AS "t3"
  INNER JOIN "partsupp" AS "t4"
    ON "t3"."p_partkey" = "t4"."ps_partkey"
) AS "t7"
WHERE
  "t7"."ps_supplycost" = (
    SELECT
      MIN("t9"."ps_supplycost") AS "Min(ps_supplycost)"
    FROM (
      SELECT
        "t8"."ps_partkey",
        "t8"."ps_supplycost"
      FROM (
        SELECT
          "t5"."ps_partkey",
          "t5"."ps_supplycost"
        FROM "partsupp" AS "t5"
        INNER JOIN "supplier" AS "t6"
          ON "t6"."s_suppkey" = "t5"."ps_suppkey"
      ) AS "t8"
      WHERE
        "t8"."ps_partkey" = "t7"."p_partkey"
    ) AS "t9"
  )