SELECT
  "t16"."s_name",
  "t16"."s_address"
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t10"."s_suppkey",
      "t10"."s_name",
      "t10"."s_address",
      "t10"."s_nationkey",
      "t10"."s_phone",
      "t10"."s_acctbal",
      "t10"."s_comment",
      "t7"."n_nationkey",
      "t7"."n_name",
      "t7"."n_regionkey",
      "t7"."n_comment"
    FROM (
      SELECT
        "t0"."s_suppkey",
        "t0"."s_name",
        "t0"."s_address",
        "t0"."s_nationkey",
        "t0"."s_phone",
        CAST("t0"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
        "t0"."s_comment"
      FROM "hive"."ibis_sf1"."supplier" AS "t0"
    ) AS "t10"
    INNER JOIN (
      SELECT
        *
      FROM "hive"."ibis_sf1"."nation" AS "t1"
    ) AS "t7"
      ON "t10"."s_nationkey" = "t7"."n_nationkey"
  ) AS "t12"
  WHERE
    "t12"."n_name" = 'CANADA'
    AND "t12"."s_suppkey" IN (
      SELECT
        "t14"."ps_suppkey"
      FROM (
        SELECT
          *
        FROM (
          SELECT
            "t2"."ps_partkey",
            "t2"."ps_suppkey",
            "t2"."ps_availqty",
            CAST("t2"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
            "t2"."ps_comment"
          FROM "hive"."ibis_sf1"."partsupp" AS "t2"
        ) AS "t8"
        WHERE
          "t8"."ps_partkey" IN (
            SELECT
              "t3"."p_partkey"
            FROM "hive"."ibis_sf1"."part" AS "t3"
            WHERE
              "t3"."p_name" LIKE 'forest%'
          )
          AND "t8"."ps_availqty" > (
            (
              SELECT
                SUM("t11"."l_quantity") AS "Sum(l_quantity)"
              FROM (
                SELECT
                  "t4"."l_orderkey",
                  "t4"."l_partkey",
                  "t4"."l_suppkey",
                  "t4"."l_linenumber",
                  CAST("t4"."l_quantity" AS DECIMAL(15, 2)) AS "l_quantity",
                  CAST("t4"."l_extendedprice" AS DECIMAL(15, 2)) AS "l_extendedprice",
                  CAST("t4"."l_discount" AS DECIMAL(15, 2)) AS "l_discount",
                  CAST("t4"."l_tax" AS DECIMAL(15, 2)) AS "l_tax",
                  "t4"."l_returnflag",
                  "t4"."l_linestatus",
                  "t4"."l_shipdate",
                  "t4"."l_commitdate",
                  "t4"."l_receiptdate",
                  "t4"."l_shipinstruct",
                  "t4"."l_shipmode",
                  "t4"."l_comment"
                FROM "hive"."ibis_sf1"."lineitem" AS "t4"
                WHERE
                  "t4"."l_partkey" = "t8"."ps_partkey"
                  AND "t4"."l_suppkey" = "t8"."ps_suppkey"
                  AND "t4"."l_shipdate" >= FROM_ISO8601_DATE('1994-01-01')
                  AND "t4"."l_shipdate" < FROM_ISO8601_DATE('1995-01-01')
              ) AS "t11"
            ) * CAST(0.5 AS DOUBLE)
          )
      ) AS "t14"
    )
) AS "t16"
ORDER BY
  "t16"."s_name" ASC