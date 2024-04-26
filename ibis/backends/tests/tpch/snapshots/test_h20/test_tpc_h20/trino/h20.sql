SELECT
  *
FROM (
  SELECT
    "t19"."s_name",
    "t19"."s_address"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t11"."s_suppkey",
        "t11"."s_name",
        "t11"."s_address",
        "t11"."s_nationkey",
        "t11"."s_phone",
        "t11"."s_acctbal",
        "t11"."s_comment",
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
      ) AS "t11"
      INNER JOIN (
        SELECT
          "t1"."n_nationkey",
          "t1"."n_name",
          "t1"."n_regionkey",
          "t1"."n_comment"
        FROM "hive"."ibis_sf1"."nation" AS "t1"
      ) AS "t7"
        ON "t11"."s_nationkey" = "t7"."n_nationkey"
    ) AS "t14"
    WHERE
      "t14"."n_name" = 'CANADA'
      AND "t14"."s_suppkey" IN (
        SELECT
          "t17"."ps_suppkey"
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
                "t12"."p_partkey"
              FROM (
                SELECT
                  *
                FROM (
                  SELECT
                    "t3"."p_partkey",
                    "t3"."p_name",
                    "t3"."p_mfgr",
                    "t3"."p_brand",
                    "t3"."p_type",
                    "t3"."p_size",
                    "t3"."p_container",
                    CAST("t3"."p_retailprice" AS DECIMAL(15, 2)) AS "p_retailprice",
                    "t3"."p_comment"
                  FROM "hive"."ibis_sf1"."part" AS "t3"
                ) AS "t9"
                WHERE
                  "t9"."p_name" LIKE 'forest%'
              ) AS "t12"
            )
            AND "t8"."ps_availqty" > (
              (
                SELECT
                  SUM("t13"."l_quantity") AS "Sum(l_quantity)"
                FROM (
                  SELECT
                    *
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
                  ) AS "t10"
                  WHERE
                    "t10"."l_partkey" = "t8"."ps_partkey"
                    AND "t10"."l_suppkey" = "t8"."ps_suppkey"
                    AND "t10"."l_shipdate" >= FROM_ISO8601_DATE('1994-01-01')
                    AND "t10"."l_shipdate" < FROM_ISO8601_DATE('1995-01-01')
                ) AS "t13"
              ) * CAST(0.5 AS DOUBLE)
            )
        ) AS "t17"
      )
  ) AS "t19"
) AS "t20"
ORDER BY
  "t20"."s_name" ASC