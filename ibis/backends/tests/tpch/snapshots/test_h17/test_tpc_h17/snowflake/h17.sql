SELECT
  SUM("t8"."l_extendedprice") / 7.0 AS "avg_yearly"
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t2"."l_orderkey" AS "l_orderkey",
      "t2"."l_partkey" AS "l_partkey",
      "t2"."l_suppkey" AS "l_suppkey",
      "t2"."l_linenumber" AS "l_linenumber",
      "t2"."l_quantity" AS "l_quantity",
      "t2"."l_extendedprice" AS "l_extendedprice",
      "t2"."l_discount" AS "l_discount",
      "t2"."l_tax" AS "l_tax",
      "t2"."l_returnflag" AS "l_returnflag",
      "t2"."l_linestatus" AS "l_linestatus",
      "t2"."l_shipdate" AS "l_shipdate",
      "t2"."l_commitdate" AS "l_commitdate",
      "t2"."l_receiptdate" AS "l_receiptdate",
      "t2"."l_shipinstruct" AS "l_shipinstruct",
      "t2"."l_shipmode" AS "l_shipmode",
      "t2"."l_comment" AS "l_comment",
      "t3"."p_partkey" AS "p_partkey",
      "t3"."p_name" AS "p_name",
      "t3"."p_mfgr" AS "p_mfgr",
      "t3"."p_brand" AS "p_brand",
      "t3"."p_type" AS "p_type",
      "t3"."p_size" AS "p_size",
      "t3"."p_container" AS "p_container",
      "t3"."p_retailprice" AS "p_retailprice",
      "t3"."p_comment" AS "p_comment"
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
    ) AS "t2"
    INNER JOIN (
      SELECT
        "t1"."P_PARTKEY" AS "p_partkey",
        "t1"."P_NAME" AS "p_name",
        "t1"."P_MFGR" AS "p_mfgr",
        "t1"."P_BRAND" AS "p_brand",
        "t1"."P_TYPE" AS "p_type",
        "t1"."P_SIZE" AS "p_size",
        "t1"."P_CONTAINER" AS "p_container",
        "t1"."P_RETAILPRICE" AS "p_retailprice",
        "t1"."P_COMMENT" AS "p_comment"
      FROM "PART" AS "t1"
    ) AS "t3"
      ON "t3"."p_partkey" = "t2"."l_partkey"
  ) AS "t5"
  WHERE
    (
      "t5"."p_brand" = 'Brand#23'
    )
    AND (
      "t5"."p_container" = 'MED BOX'
    )
    AND (
      "t5"."l_quantity" < (
        (
          SELECT
            AVG("t6"."l_quantity") AS "Mean(l_quantity)"
          FROM (
            SELECT
              *
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
            ) AS "t2"
            WHERE
              (
                "t2"."l_partkey" = "t5"."p_partkey"
              )
          ) AS "t6"
        ) * 0.2
      )
    )
) AS "t8"