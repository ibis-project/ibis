WITH "t7" AS (
  SELECT
    *
  FROM (
    SELECT
      "t3"."ps_partkey",
      "t3"."ps_suppkey",
      "t3"."ps_availqty",
      "t3"."ps_supplycost",
      "t3"."ps_comment",
      "t4"."s_suppkey",
      "t4"."s_name",
      "t4"."s_address",
      "t4"."s_nationkey",
      "t4"."s_phone",
      "t4"."s_acctbal",
      "t4"."s_comment",
      "t5"."n_nationkey",
      "t5"."n_name",
      "t5"."n_regionkey",
      "t5"."n_comment"
    FROM "partsupp" AS "t3"
    INNER JOIN "supplier" AS "t4"
      ON "t3"."ps_suppkey" = "t4"."s_suppkey"
    INNER JOIN "nation" AS "t5"
      ON "t5"."n_nationkey" = "t4"."s_nationkey"
  ) AS "t6"
  WHERE
    "t6"."n_name" = 'GERMANY'
)
SELECT
  *
FROM (
  SELECT
    "t8"."ps_partkey",
    SUM("t8"."ps_supplycost" * "t8"."ps_availqty") AS "value"
  FROM (
    SELECT
      "t8"."ps_suppkey",
      "t8"."ps_availqty",
      "t8"."ps_supplycost",
      "t8"."ps_comment",
      "t8"."s_suppkey",
      "t8"."s_name",
      "t8"."s_address",
      "t8"."s_nationkey",
      "t8"."s_phone",
      "t8"."s_acctbal",
      "t8"."s_comment",
      "t8"."n_nationkey",
      "t8"."n_name",
      "t8"."n_regionkey",
      "t8"."n_comment",
      "t8"."ps_partkey"
    FROM "t7" AS "t8"
  ) AS t8
  GROUP BY
    "t8"."ps_partkey"
) AS "t9"
WHERE
  "t9"."value" > (
    (
      SELECT
        SUM("t8"."ps_supplycost" * "t8"."ps_availqty") AS "Sum(Multiply(ps_supplycost, ps_availqty))"
      FROM "t7" AS "t8"
    ) * 0.0001
  )
ORDER BY
  "t9"."value" DESC NULLS LAST