SELECT
  "t26"."s_acctbal",
  "t26"."s_name",
  "t26"."n_name",
  "t26"."p_partkey",
  "t26"."p_mfgr",
  "t26"."s_address",
  "t26"."s_phone",
  "t26"."s_comment"
FROM (
  SELECT
    "t10"."p_partkey",
    "t10"."p_name",
    "t10"."p_mfgr",
    "t10"."p_brand",
    "t10"."p_type",
    "t10"."p_size",
    "t10"."p_container",
    "t10"."p_retailprice",
    "t10"."p_comment",
    "t11"."ps_partkey",
    "t11"."ps_suppkey",
    "t11"."ps_availqty",
    "t11"."ps_supplycost",
    "t11"."ps_comment",
    "t13"."s_suppkey",
    "t13"."s_name",
    "t13"."s_address",
    "t13"."s_nationkey",
    "t13"."s_phone",
    "t13"."s_acctbal",
    "t13"."s_comment",
    "t15"."n_nationkey",
    "t15"."n_name",
    "t15"."n_regionkey",
    "t15"."n_comment",
    "t17"."r_regionkey",
    "t17"."r_name",
    "t17"."r_comment"
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
  ) AS "t10"
  INNER JOIN (
    SELECT
      "t1"."PS_PARTKEY" AS "ps_partkey",
      "t1"."PS_SUPPKEY" AS "ps_suppkey",
      "t1"."PS_AVAILQTY" AS "ps_availqty",
      "t1"."PS_SUPPLYCOST" AS "ps_supplycost",
      "t1"."PS_COMMENT" AS "ps_comment"
    FROM "PARTSUPP" AS "t1"
  ) AS "t11"
    ON "t10"."p_partkey" = "t11"."ps_partkey"
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
  ) AS "t13"
    ON "t13"."s_suppkey" = "t11"."ps_suppkey"
  INNER JOIN (
    SELECT
      "t3"."N_NATIONKEY" AS "n_nationkey",
      "t3"."N_NAME" AS "n_name",
      "t3"."N_REGIONKEY" AS "n_regionkey",
      "t3"."N_COMMENT" AS "n_comment"
    FROM "NATION" AS "t3"
  ) AS "t15"
    ON "t13"."s_nationkey" = "t15"."n_nationkey"
  INNER JOIN (
    SELECT
      "t4"."R_REGIONKEY" AS "r_regionkey",
      "t4"."R_NAME" AS "r_name",
      "t4"."R_COMMENT" AS "r_comment"
    FROM "REGION" AS "t4"
  ) AS "t17"
    ON "t15"."n_regionkey" = "t17"."r_regionkey"
) AS "t26"
WHERE
  "t26"."p_size" = 15
  AND "t26"."p_type" LIKE '%BRASS'
  AND "t26"."r_name" = 'EUROPE'
  AND "t26"."ps_supplycost" = (
    SELECT
      MIN("t28"."ps_supplycost") AS "Min(ps_supplycost)"
    FROM (
      SELECT
        "t27"."ps_partkey",
        "t27"."ps_suppkey",
        "t27"."ps_availqty",
        "t27"."ps_supplycost",
        "t27"."ps_comment",
        "t27"."s_suppkey",
        "t27"."s_name",
        "t27"."s_address",
        "t27"."s_nationkey",
        "t27"."s_phone",
        "t27"."s_acctbal",
        "t27"."s_comment",
        "t27"."n_nationkey",
        "t27"."n_name",
        "t27"."n_regionkey",
        "t27"."n_comment",
        "t27"."r_regionkey",
        "t27"."r_name",
        "t27"."r_comment"
      FROM (
        SELECT
          "t12"."ps_partkey",
          "t12"."ps_suppkey",
          "t12"."ps_availqty",
          "t12"."ps_supplycost",
          "t12"."ps_comment",
          "t14"."s_suppkey",
          "t14"."s_name",
          "t14"."s_address",
          "t14"."s_nationkey",
          "t14"."s_phone",
          "t14"."s_acctbal",
          "t14"."s_comment",
          "t16"."n_nationkey",
          "t16"."n_name",
          "t16"."n_regionkey",
          "t16"."n_comment",
          "t18"."r_regionkey",
          "t18"."r_name",
          "t18"."r_comment"
        FROM (
          SELECT
            "t1"."PS_PARTKEY" AS "ps_partkey",
            "t1"."PS_SUPPKEY" AS "ps_suppkey",
            "t1"."PS_AVAILQTY" AS "ps_availqty",
            "t1"."PS_SUPPLYCOST" AS "ps_supplycost",
            "t1"."PS_COMMENT" AS "ps_comment"
          FROM "PARTSUPP" AS "t1"
        ) AS "t12"
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
        ) AS "t14"
          ON "t14"."s_suppkey" = "t12"."ps_suppkey"
        INNER JOIN (
          SELECT
            "t3"."N_NATIONKEY" AS "n_nationkey",
            "t3"."N_NAME" AS "n_name",
            "t3"."N_REGIONKEY" AS "n_regionkey",
            "t3"."N_COMMENT" AS "n_comment"
          FROM "NATION" AS "t3"
        ) AS "t16"
          ON "t14"."s_nationkey" = "t16"."n_nationkey"
        INNER JOIN (
          SELECT
            "t4"."R_REGIONKEY" AS "r_regionkey",
            "t4"."R_NAME" AS "r_name",
            "t4"."R_COMMENT" AS "r_comment"
          FROM "REGION" AS "t4"
        ) AS "t18"
          ON "t16"."n_regionkey" = "t18"."r_regionkey"
      ) AS "t27"
      WHERE
        "t27"."r_name" = 'EUROPE' AND "t26"."p_partkey" = "t27"."ps_partkey"
    ) AS "t28"
  )
ORDER BY
  "t26"."s_acctbal" DESC NULLS LAST,
  "t26"."n_name" ASC,
  "t26"."s_name" ASC,
  "t26"."p_partkey" ASC
LIMIT 100