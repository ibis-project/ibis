SELECT
  SUM("t8"."l_extendedprice" * (
    1 - "t8"."l_discount"
  )) AS "revenue"
FROM (
  SELECT
    "t7"."l_orderkey",
    "t7"."l_partkey",
    "t7"."l_suppkey",
    "t7"."l_linenumber",
    "t7"."l_quantity",
    "t7"."l_extendedprice",
    "t7"."l_discount",
    "t7"."l_tax",
    "t7"."l_returnflag",
    "t7"."l_linestatus",
    "t7"."l_shipdate",
    "t7"."l_commitdate",
    "t7"."l_receiptdate",
    "t7"."l_shipinstruct",
    "t7"."l_shipmode",
    "t7"."l_comment",
    "t7"."p_partkey",
    "t7"."p_name",
    "t7"."p_mfgr",
    "t7"."p_brand",
    "t7"."p_type",
    "t7"."p_size",
    "t7"."p_container",
    "t7"."p_retailprice",
    "t7"."p_comment"
  FROM (
    SELECT
      "t4"."l_orderkey",
      "t4"."l_partkey",
      "t4"."l_suppkey",
      "t4"."l_linenumber",
      "t4"."l_quantity",
      "t4"."l_extendedprice",
      "t4"."l_discount",
      "t4"."l_tax",
      "t4"."l_returnflag",
      "t4"."l_linestatus",
      "t4"."l_shipdate",
      "t4"."l_commitdate",
      "t4"."l_receiptdate",
      "t4"."l_shipinstruct",
      "t4"."l_shipmode",
      "t4"."l_comment",
      "t5"."p_partkey",
      "t5"."p_name",
      "t5"."p_mfgr",
      "t5"."p_brand",
      "t5"."p_type",
      "t5"."p_size",
      "t5"."p_container",
      "t5"."p_retailprice",
      "t5"."p_comment"
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
    ) AS "t4"
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
    ) AS "t5"
      ON "t5"."p_partkey" = "t4"."l_partkey"
  ) AS "t7"
  WHERE
    (
      (
        (
          (
            (
              (
                (
                  (
                    "t7"."p_brand" = 'Brand#12'
                  )
                  AND "t7"."p_container" IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                )
                AND (
                  "t7"."l_quantity" >= 1
                )
              )
              AND (
                "t7"."l_quantity" <= 11
              )
            )
            AND "t7"."p_size" BETWEEN 1 AND 5
          )
          AND "t7"."l_shipmode" IN ('AIR', 'AIR REG')
        )
        AND (
          "t7"."l_shipinstruct" = 'DELIVER IN PERSON'
        )
      )
      OR (
        (
          (
            (
              (
                (
                  (
                    "t7"."p_brand" = 'Brand#23'
                  )
                  AND "t7"."p_container" IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                )
                AND (
                  "t7"."l_quantity" >= 10
                )
              )
              AND (
                "t7"."l_quantity" <= 20
              )
            )
            AND "t7"."p_size" BETWEEN 1 AND 10
          )
          AND "t7"."l_shipmode" IN ('AIR', 'AIR REG')
        )
        AND (
          "t7"."l_shipinstruct" = 'DELIVER IN PERSON'
        )
      )
    )
    OR (
      (
        (
          (
            (
              (
                (
                  "t7"."p_brand" = 'Brand#34'
                )
                AND "t7"."p_container" IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
              )
              AND (
                "t7"."l_quantity" >= 20
              )
            )
            AND (
              "t7"."l_quantity" <= 30
            )
          )
          AND "t7"."p_size" BETWEEN 1 AND 15
        )
        AND "t7"."l_shipmode" IN ('AIR', 'AIR REG')
      )
      AND (
        "t7"."l_shipinstruct" = 'DELIVER IN PERSON'
      )
    )
) AS "t8"