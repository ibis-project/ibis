SELECT
  "t20"."s_name" AS "s_name",
  "t20"."numwait" AS "numwait"
FROM (
  SELECT
    "t19"."s_name" AS "s_name",
    COUNT(*) AS "numwait"
  FROM (
    SELECT
      "t16"."l1_orderkey" AS "l1_orderkey",
      "t16"."o_orderstatus" AS "o_orderstatus",
      "t16"."l_receiptdate" AS "l_receiptdate",
      "t16"."l_commitdate" AS "l_commitdate",
      "t16"."l1_suppkey" AS "l1_suppkey",
      "t16"."s_name" AS "s_name",
      "t16"."n_name" AS "n_name"
    FROM (
      SELECT
        "t8"."l_orderkey" AS "l1_orderkey",
        "t11"."o_orderstatus" AS "o_orderstatus",
        "t8"."l_receiptdate" AS "l_receiptdate",
        "t8"."l_commitdate" AS "l_commitdate",
        "t8"."l_suppkey" AS "l1_suppkey",
        "t4"."s_name" AS "s_name",
        "t12"."n_name" AS "n_name"
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
      ) AS "t4"
      INNER JOIN (
        SELECT
          "t1"."L_ORDERKEY" AS "l_orderkey",
          "t1"."L_PARTKEY" AS "l_partkey",
          "t1"."L_SUPPKEY" AS "l_suppkey",
          "t1"."L_LINENUMBER" AS "l_linenumber",
          "t1"."L_QUANTITY" AS "l_quantity",
          "t1"."L_EXTENDEDPRICE" AS "l_extendedprice",
          "t1"."L_DISCOUNT" AS "l_discount",
          "t1"."L_TAX" AS "l_tax",
          "t1"."L_RETURNFLAG" AS "l_returnflag",
          "t1"."L_LINESTATUS" AS "l_linestatus",
          "t1"."L_SHIPDATE" AS "l_shipdate",
          "t1"."L_COMMITDATE" AS "l_commitdate",
          "t1"."L_RECEIPTDATE" AS "l_receiptdate",
          "t1"."L_SHIPINSTRUCT" AS "l_shipinstruct",
          "t1"."L_SHIPMODE" AS "l_shipmode",
          "t1"."L_COMMENT" AS "l_comment"
        FROM "LINEITEM" AS "t1"
      ) AS "t8"
        ON "t4"."s_suppkey" = "t8"."l_suppkey"
      INNER JOIN (
        SELECT
          "t2"."O_ORDERKEY" AS "o_orderkey",
          "t2"."O_CUSTKEY" AS "o_custkey",
          "t2"."O_ORDERSTATUS" AS "o_orderstatus",
          "t2"."O_TOTALPRICE" AS "o_totalprice",
          "t2"."O_ORDERDATE" AS "o_orderdate",
          "t2"."O_ORDERPRIORITY" AS "o_orderpriority",
          "t2"."O_CLERK" AS "o_clerk",
          "t2"."O_SHIPPRIORITY" AS "o_shippriority",
          "t2"."O_COMMENT" AS "o_comment"
        FROM "ORDERS" AS "t2"
      ) AS "t11"
        ON "t11"."o_orderkey" = "t8"."l_orderkey"
      INNER JOIN (
        SELECT
          "t3"."N_NATIONKEY" AS "n_nationkey",
          "t3"."N_NAME" AS "n_name",
          "t3"."N_REGIONKEY" AS "n_regionkey",
          "t3"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t3"
      ) AS "t12"
        ON "t4"."s_nationkey" = "t12"."n_nationkey"
    ) AS "t16"
    WHERE
      "t16"."o_orderstatus" = 'F'
      AND "t16"."l_receiptdate" > "t16"."l_commitdate"
      AND "t16"."n_name" = 'SAUDI ARABIA'
      AND EXISTS(
        (
          SELECT
            1 AS "1"
          FROM (
            SELECT
              "t1"."L_ORDERKEY" AS "l_orderkey",
              "t1"."L_PARTKEY" AS "l_partkey",
              "t1"."L_SUPPKEY" AS "l_suppkey",
              "t1"."L_LINENUMBER" AS "l_linenumber",
              "t1"."L_QUANTITY" AS "l_quantity",
              "t1"."L_EXTENDEDPRICE" AS "l_extendedprice",
              "t1"."L_DISCOUNT" AS "l_discount",
              "t1"."L_TAX" AS "l_tax",
              "t1"."L_RETURNFLAG" AS "l_returnflag",
              "t1"."L_LINESTATUS" AS "l_linestatus",
              "t1"."L_SHIPDATE" AS "l_shipdate",
              "t1"."L_COMMITDATE" AS "l_commitdate",
              "t1"."L_RECEIPTDATE" AS "l_receiptdate",
              "t1"."L_SHIPINSTRUCT" AS "l_shipinstruct",
              "t1"."L_SHIPMODE" AS "l_shipmode",
              "t1"."L_COMMENT" AS "l_comment"
            FROM "LINEITEM" AS "t1"
          ) AS "t9"
          WHERE
            (
              "t9"."l_orderkey" = "t16"."l1_orderkey"
            )
            AND (
              "t9"."l_suppkey" <> "t16"."l1_suppkey"
            )
        )
      )
      AND NOT (
        EXISTS(
          (
            SELECT
              1 AS "1"
            FROM (
              SELECT
                "t1"."L_ORDERKEY" AS "l_orderkey",
                "t1"."L_PARTKEY" AS "l_partkey",
                "t1"."L_SUPPKEY" AS "l_suppkey",
                "t1"."L_LINENUMBER" AS "l_linenumber",
                "t1"."L_QUANTITY" AS "l_quantity",
                "t1"."L_EXTENDEDPRICE" AS "l_extendedprice",
                "t1"."L_DISCOUNT" AS "l_discount",
                "t1"."L_TAX" AS "l_tax",
                "t1"."L_RETURNFLAG" AS "l_returnflag",
                "t1"."L_LINESTATUS" AS "l_linestatus",
                "t1"."L_SHIPDATE" AS "l_shipdate",
                "t1"."L_COMMITDATE" AS "l_commitdate",
                "t1"."L_RECEIPTDATE" AS "l_receiptdate",
                "t1"."L_SHIPINSTRUCT" AS "l_shipinstruct",
                "t1"."L_SHIPMODE" AS "l_shipmode",
                "t1"."L_COMMENT" AS "l_comment"
              FROM "LINEITEM" AS "t1"
            ) AS "t10"
            WHERE
              (
                (
                  "t10"."l_orderkey" = "t16"."l1_orderkey"
                )
                AND (
                  "t10"."l_suppkey" <> "t16"."l1_suppkey"
                )
              )
              AND (
                "t10"."l_receiptdate" > "t10"."l_commitdate"
              )
          )
        )
      )
  ) AS "t19"
  GROUP BY
    1
) AS "t20"
ORDER BY
  "t20"."numwait" DESC NULLS LAST,
  "t20"."s_name" ASC
LIMIT 100