SELECT
  "t24"."s_acctbal" AS "s_acctbal",
  "t24"."s_name" AS "s_name",
  "t24"."n_name" AS "n_name",
  "t24"."p_partkey" AS "p_partkey",
  "t24"."p_mfgr" AS "p_mfgr",
  "t24"."s_address" AS "s_address",
  "t24"."s_phone" AS "s_phone",
  "t24"."s_comment" AS "s_comment"
FROM (
  SELECT
    "t5"."p_partkey" AS "p_partkey",
    "t5"."p_name" AS "p_name",
    "t5"."p_mfgr" AS "p_mfgr",
    "t5"."p_brand" AS "p_brand",
    "t5"."p_type" AS "p_type",
    "t5"."p_size" AS "p_size",
    "t5"."p_container" AS "p_container",
    "t5"."p_retailprice" AS "p_retailprice",
    "t5"."p_comment" AS "p_comment",
    "t10"."ps_partkey" AS "ps_partkey",
    "t10"."ps_suppkey" AS "ps_suppkey",
    "t10"."ps_availqty" AS "ps_availqty",
    "t10"."ps_supplycost" AS "ps_supplycost",
    "t10"."ps_comment" AS "ps_comment",
    "t11"."s_suppkey" AS "s_suppkey",
    "t11"."s_name" AS "s_name",
    "t11"."s_address" AS "s_address",
    "t11"."s_nationkey" AS "s_nationkey",
    "t11"."s_phone" AS "s_phone",
    "t11"."s_acctbal" AS "s_acctbal",
    "t11"."s_comment" AS "s_comment",
    "t13"."n_nationkey" AS "n_nationkey",
    "t13"."n_name" AS "n_name",
    "t13"."n_regionkey" AS "n_regionkey",
    "t13"."n_comment" AS "n_comment",
    "t15"."r_regionkey" AS "r_regionkey",
    "t15"."r_name" AS "r_name",
    "t15"."r_comment" AS "r_comment"
  FROM (
    SELECT
      "t0"."P_PARTKEY" AS "p_partkey",
      "t0"."P_NAME" AS "p_name",
      "t0"."P_MFGR" AS "p_mfgr",
      "t0"."P_BRAND" AS "p_brand",
      "t0"."P_TYPE" AS "p_type",
      "t0"."P_SIZE" AS "p_size",
      "t0"."P_CONTAINER" AS "p_container",
      "t0"."P_RETAILPRICE" AS "p_retailprice",
      "t0"."P_COMMENT" AS "p_comment"
    FROM "PART" AS "t0"
  ) AS "t5"
  INNER JOIN (
    SELECT
      "t1"."PS_PARTKEY" AS "ps_partkey",
      "t1"."PS_SUPPKEY" AS "ps_suppkey",
      "t1"."PS_AVAILQTY" AS "ps_availqty",
      "t1"."PS_SUPPLYCOST" AS "ps_supplycost",
      "t1"."PS_COMMENT" AS "ps_comment"
    FROM "PARTSUPP" AS "t1"
  ) AS "t10"
    ON "t5"."p_partkey" = "t10"."ps_partkey"
  INNER JOIN (
    SELECT
      "t2"."S_SUPPKEY" AS "s_suppkey",
      "t2"."S_NAME" AS "s_name",
      "t2"."S_ADDRESS" AS "s_address",
      "t2"."S_NATIONKEY" AS "s_nationkey",
      "t2"."S_PHONE" AS "s_phone",
      "t2"."S_ACCTBAL" AS "s_acctbal",
      "t2"."S_COMMENT" AS "s_comment"
    FROM "SUPPLIER" AS "t2"
  ) AS "t11"
    ON "t11"."s_suppkey" = "t10"."ps_suppkey"
  INNER JOIN (
    SELECT
      "t3"."N_NATIONKEY" AS "n_nationkey",
      "t3"."N_NAME" AS "n_name",
      "t3"."N_REGIONKEY" AS "n_regionkey",
      "t3"."N_COMMENT" AS "n_comment"
    FROM "NATION" AS "t3"
  ) AS "t13"
    ON "t11"."s_nationkey" = "t13"."n_nationkey"
  INNER JOIN (
    SELECT
      "t4"."R_REGIONKEY" AS "r_regionkey",
      "t4"."R_NAME" AS "r_name",
      "t4"."R_COMMENT" AS "r_comment"
    FROM "REGION" AS "t4"
  ) AS "t15"
    ON "t13"."n_regionkey" = "t15"."r_regionkey"
) AS "t24"
WHERE
  "t24"."p_size" = 15
  AND "t24"."p_type" LIKE '%BRASS'
  AND "t24"."r_name" = 'EUROPE'
  AND "t24"."ps_supplycost" = (
    SELECT
      MIN("t26"."ps_supplycost") AS "Min(ps_supplycost)"
    FROM (
      SELECT
        "t25"."ps_partkey" AS "ps_partkey",
        "t25"."ps_suppkey" AS "ps_suppkey",
        "t25"."ps_availqty" AS "ps_availqty",
        "t25"."ps_supplycost" AS "ps_supplycost",
        "t25"."ps_comment" AS "ps_comment",
        "t25"."s_suppkey" AS "s_suppkey",
        "t25"."s_name" AS "s_name",
        "t25"."s_address" AS "s_address",
        "t25"."s_nationkey" AS "s_nationkey",
        "t25"."s_phone" AS "s_phone",
        "t25"."s_acctbal" AS "s_acctbal",
        "t25"."s_comment" AS "s_comment",
        "t25"."n_nationkey" AS "n_nationkey",
        "t25"."n_name" AS "n_name",
        "t25"."n_regionkey" AS "n_regionkey",
        "t25"."n_comment" AS "n_comment",
        "t25"."r_regionkey" AS "r_regionkey",
        "t25"."r_name" AS "r_name",
        "t25"."r_comment" AS "r_comment"
      FROM (
        SELECT
          "t6"."ps_partkey" AS "ps_partkey",
          "t6"."ps_suppkey" AS "ps_suppkey",
          "t6"."ps_availqty" AS "ps_availqty",
          "t6"."ps_supplycost" AS "ps_supplycost",
          "t6"."ps_comment" AS "ps_comment",
          "t12"."s_suppkey" AS "s_suppkey",
          "t12"."s_name" AS "s_name",
          "t12"."s_address" AS "s_address",
          "t12"."s_nationkey" AS "s_nationkey",
          "t12"."s_phone" AS "s_phone",
          "t12"."s_acctbal" AS "s_acctbal",
          "t12"."s_comment" AS "s_comment",
          "t14"."n_nationkey" AS "n_nationkey",
          "t14"."n_name" AS "n_name",
          "t14"."n_regionkey" AS "n_regionkey",
          "t14"."n_comment" AS "n_comment",
          "t16"."r_regionkey" AS "r_regionkey",
          "t16"."r_name" AS "r_name",
          "t16"."r_comment" AS "r_comment"
        FROM (
          SELECT
            "t1"."PS_PARTKEY" AS "ps_partkey",
            "t1"."PS_SUPPKEY" AS "ps_suppkey",
            "t1"."PS_AVAILQTY" AS "ps_availqty",
            "t1"."PS_SUPPLYCOST" AS "ps_supplycost",
            "t1"."PS_COMMENT" AS "ps_comment"
          FROM "PARTSUPP" AS "t1"
        ) AS "t6"
        INNER JOIN (
          SELECT
            "t2"."S_SUPPKEY" AS "s_suppkey",
            "t2"."S_NAME" AS "s_name",
            "t2"."S_ADDRESS" AS "s_address",
            "t2"."S_NATIONKEY" AS "s_nationkey",
            "t2"."S_PHONE" AS "s_phone",
            "t2"."S_ACCTBAL" AS "s_acctbal",
            "t2"."S_COMMENT" AS "s_comment"
          FROM "SUPPLIER" AS "t2"
        ) AS "t12"
          ON "t12"."s_suppkey" = "t6"."ps_suppkey"
        INNER JOIN (
          SELECT
            "t3"."N_NATIONKEY" AS "n_nationkey",
            "t3"."N_NAME" AS "n_name",
            "t3"."N_REGIONKEY" AS "n_regionkey",
            "t3"."N_COMMENT" AS "n_comment"
          FROM "NATION" AS "t3"
        ) AS "t14"
          ON "t12"."s_nationkey" = "t14"."n_nationkey"
        INNER JOIN (
          SELECT
            "t4"."R_REGIONKEY" AS "r_regionkey",
            "t4"."R_NAME" AS "r_name",
            "t4"."R_COMMENT" AS "r_comment"
          FROM "REGION" AS "t4"
        ) AS "t16"
          ON "t14"."n_regionkey" = "t16"."r_regionkey"
      ) AS "t25"
      WHERE
        "t25"."r_name" = 'EUROPE' AND "t24"."p_partkey" = "t25"."ps_partkey"
    ) AS "t26"
  )
ORDER BY
  "t24"."s_acctbal" DESC NULLS LAST,
  "t24"."n_name" ASC,
  "t24"."s_name" ASC,
  "t24"."p_partkey" ASC
LIMIT 100