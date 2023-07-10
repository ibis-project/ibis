SELECT
  "t12"."s_name" AS "s_name",
  "t12"."s_address" AS "s_address"
FROM (
  SELECT
    "t5"."s_suppkey" AS "s_suppkey",
    "t5"."s_name" AS "s_name",
    "t5"."s_address" AS "s_address",
    "t5"."s_nationkey" AS "s_nationkey",
    "t5"."s_phone" AS "s_phone",
    "t5"."s_acctbal" AS "s_acctbal",
    "t5"."s_comment" AS "s_comment",
    "t7"."n_nationkey" AS "n_nationkey",
    "t7"."n_name" AS "n_name",
    "t7"."n_regionkey" AS "n_regionkey",
    "t7"."n_comment" AS "n_comment"
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
      "t2"."N_NATIONKEY" AS "n_nationkey",
      "t2"."N_NAME" AS "n_name",
      "t2"."N_REGIONKEY" AS "n_regionkey",
      "t2"."N_COMMENT" AS "n_comment"
    FROM "NATION" AS "t2"
  ) AS "t7"
    ON "t5"."s_nationkey" = "t7"."n_nationkey"
) AS "t12"
WHERE
  "t12"."n_name" = 'CANADA'
  AND "t12"."s_suppkey" IN ((
    SELECT
      "t1"."PS_SUPPKEY" AS "ps_suppkey"
    FROM "PARTSUPP" AS "t1"
    WHERE
      "t1"."PS_PARTKEY" IN ((
        SELECT
          "t3"."P_PARTKEY" AS "p_partkey"
        FROM "PART" AS "t3"
        WHERE
          "t3"."P_NAME" LIKE 'forest%'
      ))
      AND "t1"."PS_AVAILQTY" > (
        (
          SELECT
            SUM("t9"."l_quantity") AS "Sum(l_quantity)"
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
            WHERE
              "t4"."L_PARTKEY" = "t1"."PS_PARTKEY"
              AND "t4"."L_SUPPKEY" = "t1"."PS_SUPPKEY"
              AND "t4"."L_SHIPDATE" >= DATEFROMPARTS(1994, 1, 1)
              AND "t4"."L_SHIPDATE" < DATEFROMPARTS(1995, 1, 1)
          ) AS "t9"
        ) * 0.5
      )
  ))
ORDER BY
  "t12"."s_name" ASC