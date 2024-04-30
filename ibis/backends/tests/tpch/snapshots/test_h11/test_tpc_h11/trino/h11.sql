WITH "t10" AS (
  SELECT
    *
  FROM (
    SELECT
      "t7"."ps_partkey",
      "t7"."ps_suppkey",
      "t7"."ps_availqty",
      "t7"."ps_supplycost",
      "t7"."ps_comment",
      "t8"."s_suppkey",
      "t8"."s_name",
      "t8"."s_address",
      "t8"."s_nationkey",
      "t8"."s_phone",
      "t8"."s_acctbal",
      "t8"."s_comment",
      "t6"."n_nationkey",
      "t6"."n_name",
      "t6"."n_regionkey",
      "t6"."n_comment"
    FROM (
      SELECT
        "t0"."ps_partkey",
        "t0"."ps_suppkey",
        "t0"."ps_availqty",
        CAST("t0"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
        "t0"."ps_comment"
      FROM "hive"."ibis_sf1"."partsupp" AS "t0"
    ) AS "t7"
    INNER JOIN (
      SELECT
        "t1"."s_suppkey",
        "t1"."s_name",
        "t1"."s_address",
        "t1"."s_nationkey",
        "t1"."s_phone",
        CAST("t1"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
        "t1"."s_comment"
      FROM "hive"."ibis_sf1"."supplier" AS "t1"
    ) AS "t8"
      ON "t7"."ps_suppkey" = "t8"."s_suppkey"
    INNER JOIN (
      SELECT
        *
      FROM "hive"."ibis_sf1"."nation" AS "t2"
    ) AS "t6"
      ON "t6"."n_nationkey" = "t8"."s_nationkey"
  ) AS "t9"
  WHERE
    "t9"."n_name" = 'GERMANY'
)
SELECT
  *
FROM (
  SELECT
    "t11"."ps_partkey",
    SUM("t11"."ps_supplycost" * "t11"."ps_availqty") AS "value"
  FROM "t10" AS "t11"
  GROUP BY
    1
) AS "t12"
WHERE
  "t12"."value" > (
    (
      SELECT
        SUM("t11"."ps_supplycost" * "t11"."ps_availqty") AS "Sum(Multiply(ps_supplycost, ps_availqty))"
      FROM "t10" AS "t11"
    ) * CAST(0.0001 AS DOUBLE)
  )
ORDER BY
  "t12"."value" DESC