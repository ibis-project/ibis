SELECT
  *
FROM (
  SELECT
    "t10"."p_brand" AS "p_brand",
    "t10"."p_type" AS "p_type",
    "t10"."p_size" AS "p_size",
    COUNT(DISTINCT "t10"."ps_suppkey") AS "supplier_cnt"
  FROM (
    SELECT
      *
    FROM (
      SELECT
        "t3"."ps_partkey" AS "ps_partkey",
        "t3"."ps_suppkey" AS "ps_suppkey",
        "t3"."ps_availqty" AS "ps_availqty",
        "t3"."ps_supplycost" AS "ps_supplycost",
        "t3"."ps_comment" AS "ps_comment",
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
          "t0"."PS_PARTKEY" AS "ps_partkey",
          "t0"."PS_SUPPKEY" AS "ps_suppkey",
          "t0"."PS_AVAILQTY" AS "ps_availqty",
          "t0"."PS_SUPPLYCOST" AS "ps_supplycost",
          "t0"."PS_COMMENT" AS "ps_comment"
        FROM "PARTSUPP" AS "t0"
      ) AS "t3"
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
        ON "t4"."p_partkey" = "t3"."ps_partkey"
    ) AS "t8"
    WHERE
      (
        "t8"."p_brand" <> 'Brand#45'
      )
      AND NOT (
        "t8"."p_type" LIKE 'MEDIUM POLISHED%'
      )
      AND "t8"."p_size" IN (49, 14, 23, 45, 19, 3, 36, 9)
      AND NOT (
        "t8"."ps_suppkey" IN ((
          SELECT
            "t7"."s_suppkey" AS "s_suppkey"
          FROM (
            SELECT
              *
            FROM (
              SELECT
                "t2"."S_SUPPKEY" AS "s_suppkey",
                "t2"."S_NAME" AS "s_name",
                "t2"."S_ADDRESS" AS "s_address",
                "t2"."S_NATIONKEY" AS "s_nationkey",
                "t2"."S_PHONE" AS "s_phone",
                "t2"."S_ACCTBAL" AS "s_acctbal",
                "t2"."S_COMMENT" AS "s_comment"
              FROM "SUPPLIER" AS "t2"
            ) AS "t5"
            WHERE
              "t5"."s_comment" LIKE '%Customer%Complaints%'
          ) AS "t7"
        ))
      )
  ) AS "t10"
  GROUP BY
    1,
    2,
    3
) AS "t11"
ORDER BY
  "t11"."supplier_cnt" DESC NULLS LAST,
  "t11"."p_brand" ASC,
  "t11"."p_type" ASC,
  "t11"."p_size" ASC