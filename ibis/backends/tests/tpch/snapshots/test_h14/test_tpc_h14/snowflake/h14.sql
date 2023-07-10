SELECT
  (
    SUM(
      IFF("t7"."p_type" LIKE 'PROMO%', "t7"."l_extendedprice" * (
        1 - "t7"."l_discount"
      ), 0)
    ) * 100
  ) / SUM("t7"."l_extendedprice" * (
    1 - "t7"."l_discount"
  )) AS "promo_revenue"
FROM (
  SELECT
    "t6"."l_orderkey" AS "l_orderkey",
    "t6"."l_partkey" AS "l_partkey",
    "t6"."l_suppkey" AS "l_suppkey",
    "t6"."l_linenumber" AS "l_linenumber",
    "t6"."l_quantity" AS "l_quantity",
    "t6"."l_extendedprice" AS "l_extendedprice",
    "t6"."l_discount" AS "l_discount",
    "t6"."l_tax" AS "l_tax",
    "t6"."l_returnflag" AS "l_returnflag",
    "t6"."l_linestatus" AS "l_linestatus",
    "t6"."l_shipdate" AS "l_shipdate",
    "t6"."l_commitdate" AS "l_commitdate",
    "t6"."l_receiptdate" AS "l_receiptdate",
    "t6"."l_shipinstruct" AS "l_shipinstruct",
    "t6"."l_shipmode" AS "l_shipmode",
    "t6"."l_comment" AS "l_comment",
    "t6"."p_partkey" AS "p_partkey",
    "t6"."p_name" AS "p_name",
    "t6"."p_mfgr" AS "p_mfgr",
    "t6"."p_brand" AS "p_brand",
    "t6"."p_type" AS "p_type",
    "t6"."p_size" AS "p_size",
    "t6"."p_container" AS "p_container",
    "t6"."p_retailprice" AS "p_retailprice",
    "t6"."p_comment" AS "p_comment"
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
      "t4"."p_partkey" AS "p_partkey",
      "t4"."p_name" AS "p_name",
      "t4"."p_mfgr" AS "p_mfgr",
      "t4"."p_brand" AS "p_brand",
      "t4"."p_type" AS "p_type",
      "t4"."p_size" AS "p_size",
      "t4"."p_container" AS "p_container",
      "t4"."p_retailprice" AS "p_retailprice",
      "t4"."p_comment" AS "p_comment"
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
    ) AS "t4"
      ON "t2"."l_partkey" = "t4"."p_partkey"
  ) AS "t6"
  WHERE
    "t6"."l_shipdate" >= DATEFROMPARTS(1995, 9, 1)
    AND "t6"."l_shipdate" < DATEFROMPARTS(1995, 10, 1)
) AS "t7"