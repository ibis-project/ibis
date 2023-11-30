SELECT
  *
FROM (
  SELECT
    "t18"."s_name" AS "s_name",
    "t18"."s_address" AS "s_address"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t5"."s_suppkey" AS "s_suppkey",
        "t5"."s_name" AS "s_name",
        "t5"."s_address" AS "s_address",
        "t5"."s_nationkey" AS "s_nationkey",
        "t5"."s_phone" AS "s_phone",
        "t5"."s_acctbal" AS "s_acctbal",
        "t5"."s_comment" AS "s_comment",
        "t6"."n_nationkey" AS "n_nationkey",
        "t6"."n_name" AS "n_name",
        "t6"."n_regionkey" AS "n_regionkey",
        "t6"."n_comment" AS "n_comment"
      FROM (
        SELECT
          "t0"."S_SUPPKEY" AS "s_suppkey",
          "t0"."S_NAME" AS "s_name",
          "t0"."S_ADDRESS" AS "s_address",
          "t0"."S_NATIONKEY" AS "s_nationkey",
          "t0"."S_PHONE" AS "s_phone",
          "t0"."S_ACCTBAL" AS "s_acctbal",
          "t0"."S_COMMENT" AS "s_comment"
        FROM "SUPPLIER" AS "t0"
      ) AS "t5"
      INNER JOIN (
        SELECT
          "t1"."N_NATIONKEY" AS "n_nationkey",
          "t1"."N_NAME" AS "n_name",
          "t1"."N_REGIONKEY" AS "n_regionkey",
          "t1"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t1"
      ) AS "t6"
        ON "t5"."s_nationkey" = "t6"."n_nationkey"
    ) AS "t13"
    WHERE
      (
        "t13"."n_name" = 'CANADA'
      )
      AND "t13"."s_suppkey" IN ((
        SELECT
          "t16"."ps_suppkey" AS "ps_suppkey"
        FROM (
          SELECT
            *
          FROM (
            SELECT
              "t2"."PS_PARTKEY" AS "ps_partkey",
              "t2"."PS_SUPPKEY" AS "ps_suppkey",
              "t2"."PS_AVAILQTY" AS "ps_availqty",
              "t2"."PS_SUPPLYCOST" AS "ps_supplycost",
              "t2"."PS_COMMENT" AS "ps_comment"
            FROM "PARTSUPP" AS "t2"
          ) AS "t7"
          WHERE
            "t7"."ps_partkey" IN ((
              SELECT
                "t11"."p_partkey" AS "p_partkey"
              FROM (
                SELECT
                  *
                FROM (
                  SELECT
                    "t3"."P_PARTKEY" AS "p_partkey",
                    "t3"."P_NAME" AS "p_name",
                    "t3"."P_MFGR" AS "p_mfgr",
                    "t3"."P_BRAND" AS "p_brand",
                    "t3"."P_TYPE" AS "p_type",
                    "t3"."P_SIZE" AS "p_size",
                    "t3"."P_CONTAINER" AS "p_container",
                    "t3"."P_RETAILPRICE" AS "p_retailprice",
                    "t3"."P_COMMENT" AS "p_comment"
                  FROM "PART" AS "t3"
                ) AS "t8"
                WHERE
                  "t8"."p_name" LIKE 'forest%'
              ) AS "t11"
            ))
            AND (
              "t7"."ps_availqty" > (
                (
                  SELECT
                    SUM("t12"."l_quantity") AS "Sum(l_quantity)"
                  FROM (
                    SELECT
                      *
                    FROM (
                      SELECT
                        "t4"."L_ORDERKEY" AS "l_orderkey",
                        "t4"."L_PARTKEY" AS "l_partkey",
                        "t4"."L_SUPPKEY" AS "l_suppkey",
                        "t4"."L_LINENUMBER" AS "l_linenumber",
                        "t4"."L_QUANTITY" AS "l_quantity",
                        "t4"."L_EXTENDEDPRICE" AS "l_extendedprice",
                        "t4"."L_DISCOUNT" AS "l_discount",
                        "t4"."L_TAX" AS "l_tax",
                        "t4"."L_RETURNFLAG" AS "l_returnflag",
                        "t4"."L_LINESTATUS" AS "l_linestatus",
                        "t4"."L_SHIPDATE" AS "l_shipdate",
                        "t4"."L_COMMITDATE" AS "l_commitdate",
                        "t4"."L_RECEIPTDATE" AS "l_receiptdate",
                        "t4"."L_SHIPINSTRUCT" AS "l_shipinstruct",
                        "t4"."L_SHIPMODE" AS "l_shipmode",
                        "t4"."L_COMMENT" AS "l_comment"
                      FROM "LINEITEM" AS "t4"
                    ) AS "t9"
                    WHERE
                      (
                        "t9"."l_partkey" = "t7"."ps_partkey"
                      )
                      AND (
                        "t9"."l_suppkey" = "t7"."ps_suppkey"
                      )
                      AND (
                        "t9"."l_shipdate" >= DATEFROMPARTS(1994, 1, 1)
                      )
                      AND (
                        "t9"."l_shipdate" < DATEFROMPARTS(1995, 1, 1)
                      )
                  ) AS "t12"
                ) * 0.5
              )
            )
        ) AS "t16"
      ))
  ) AS "t18"
) AS "t19"
ORDER BY
  "t19"."s_name" ASC