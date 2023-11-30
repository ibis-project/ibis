SELECT
  *
FROM (
  SELECT
    "t18"."nation" AS "nation",
    "t18"."o_year" AS "o_year",
    SUM("t18"."amount") AS "sum_profit"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        (
          "t6"."l_extendedprice" * (
            1 - "t6"."l_discount"
          )
        ) - (
          "t8"."ps_supplycost" * "t6"."l_quantity"
        ) AS "amount",
        DATE_PART('year', "t10"."o_orderdate") AS "o_year",
        "t11"."n_name" AS "nation",
        "t9"."p_name" AS "p_name"
      FROM (
        SELECT
          "t0"."L_ORDERKEY" AS "l_orderkey",
          "t0"."L_PARTKEY" AS "l_partkey",
          "t0"."L_SUPPKEY" AS "l_suppkey",
          "t0"."L_LINENUMBER" AS "l_linenumber",
          "t0"."L_QUANTITY" AS "l_quantity",
          "t0"."L_EXTENDEDPRICE" AS "l_extendedprice",
          "t0"."L_DISCOUNT" AS "l_discount",
          "t0"."L_TAX" AS "l_tax",
          "t0"."L_RETURNFLAG" AS "l_returnflag",
          "t0"."L_LINESTATUS" AS "l_linestatus",
          "t0"."L_SHIPDATE" AS "l_shipdate",
          "t0"."L_COMMITDATE" AS "l_commitdate",
          "t0"."L_RECEIPTDATE" AS "l_receiptdate",
          "t0"."L_SHIPINSTRUCT" AS "l_shipinstruct",
          "t0"."L_SHIPMODE" AS "l_shipmode",
          "t0"."L_COMMENT" AS "l_comment"
        FROM "LINEITEM" AS "t0"
      ) AS "t6"
      INNER JOIN (
        SELECT
          "t1"."S_SUPPKEY" AS "s_suppkey",
          "t1"."S_NAME" AS "s_name",
          "t1"."S_ADDRESS" AS "s_address",
          "t1"."S_NATIONKEY" AS "s_nationkey",
          "t1"."S_PHONE" AS "s_phone",
          "t1"."S_ACCTBAL" AS "s_acctbal",
          "t1"."S_COMMENT" AS "s_comment"
        FROM "SUPPLIER" AS "t1"
      ) AS "t7"
        ON "t7"."s_suppkey" = "t6"."l_suppkey"
      INNER JOIN (
        SELECT
          "t2"."PS_PARTKEY" AS "ps_partkey",
          "t2"."PS_SUPPKEY" AS "ps_suppkey",
          "t2"."PS_AVAILQTY" AS "ps_availqty",
          "t2"."PS_SUPPLYCOST" AS "ps_supplycost",
          "t2"."PS_COMMENT" AS "ps_comment"
        FROM "PARTSUPP" AS "t2"
      ) AS "t8"
        ON "t8"."ps_suppkey" = "t6"."l_suppkey" AND "t8"."ps_partkey" = "t6"."l_partkey"
      INNER JOIN (
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
      ) AS "t9"
        ON "t9"."p_partkey" = "t6"."l_partkey"
      INNER JOIN (
        SELECT
          "t4"."O_ORDERKEY" AS "o_orderkey",
          "t4"."O_CUSTKEY" AS "o_custkey",
          "t4"."O_ORDERSTATUS" AS "o_orderstatus",
          "t4"."O_TOTALPRICE" AS "o_totalprice",
          "t4"."O_ORDERDATE" AS "o_orderdate",
          "t4"."O_ORDERPRIORITY" AS "o_orderpriority",
          "t4"."O_CLERK" AS "o_clerk",
          "t4"."O_SHIPPRIORITY" AS "o_shippriority",
          "t4"."O_COMMENT" AS "o_comment"
        FROM "ORDERS" AS "t4"
      ) AS "t10"
        ON "t10"."o_orderkey" = "t6"."l_orderkey"
      INNER JOIN (
        SELECT
          "t5"."N_NATIONKEY" AS "n_nationkey",
          "t5"."N_NAME" AS "n_name",
          "t5"."N_REGIONKEY" AS "n_regionkey",
          "t5"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t5"
      ) AS "t11"
        ON "t7"."s_nationkey" = "t11"."n_nationkey"
    ) AS "t17"
    WHERE
      "t17"."p_name" LIKE '%green%'
  ) AS "t18"
  GROUP BY
    1,
    2
) AS "t19"
ORDER BY
  "t19"."nation" ASC,
  "t19"."o_year" DESC NULLS LAST