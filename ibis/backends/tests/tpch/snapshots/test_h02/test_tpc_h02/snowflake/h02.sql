SELECT
  *
FROM (
  SELECT
    "t18"."s_acctbal" AS "s_acctbal",
    "t18"."s_name" AS "s_name",
    "t18"."n_name" AS "n_name",
    "t18"."p_partkey" AS "p_partkey",
    "t18"."p_mfgr" AS "p_mfgr",
    "t18"."s_address" AS "s_address",
    "t18"."s_phone" AS "s_phone",
    "t18"."s_comment" AS "s_comment"
  FROM (
    SELECT
      *
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
        "t6"."ps_partkey" AS "ps_partkey",
        "t6"."ps_suppkey" AS "ps_suppkey",
        "t6"."ps_availqty" AS "ps_availqty",
        "t6"."ps_supplycost" AS "ps_supplycost",
        "t6"."ps_comment" AS "ps_comment",
        "t7"."s_suppkey" AS "s_suppkey",
        "t7"."s_name" AS "s_name",
        "t7"."s_address" AS "s_address",
        "t7"."s_nationkey" AS "s_nationkey",
        "t7"."s_phone" AS "s_phone",
        "t7"."s_acctbal" AS "s_acctbal",
        "t7"."s_comment" AS "s_comment",
        "t8"."n_nationkey" AS "n_nationkey",
        "t8"."n_name" AS "n_name",
        "t8"."n_regionkey" AS "n_regionkey",
        "t8"."n_comment" AS "n_comment",
        "t9"."r_regionkey" AS "r_regionkey",
        "t9"."r_name" AS "r_name",
        "t9"."r_comment" AS "r_comment"
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
      ) AS "t6"
        ON "t5"."p_partkey" = "t6"."ps_partkey"
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
      ) AS "t7"
        ON "t7"."s_suppkey" = "t6"."ps_suppkey"
      INNER JOIN (
        SELECT
          "t3"."N_NATIONKEY" AS "n_nationkey",
          "t3"."N_NAME" AS "n_name",
          "t3"."N_REGIONKEY" AS "n_regionkey",
          "t3"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t3"
      ) AS "t8"
        ON "t7"."s_nationkey" = "t8"."n_nationkey"
      INNER JOIN (
        SELECT
          "t4"."R_REGIONKEY" AS "r_regionkey",
          "t4"."R_NAME" AS "r_name",
          "t4"."R_COMMENT" AS "r_comment"
        FROM "REGION" AS "t4"
      ) AS "t9"
        ON "t8"."n_regionkey" = "t9"."r_regionkey"
    ) AS "t14"
    WHERE
      (
        "t14"."p_size" = 15
      )
      AND "t14"."p_type" LIKE '%BRASS'
      AND (
        "t14"."r_name" = 'EUROPE'
      )
      AND (
        "t14"."ps_supplycost" = (
          SELECT
            MIN("t16"."ps_supplycost") AS "Min(ps_supplycost)"
          FROM (
            SELECT
              *
            FROM (
              SELECT
                "t6"."ps_partkey" AS "ps_partkey",
                "t6"."ps_suppkey" AS "ps_suppkey",
                "t6"."ps_availqty" AS "ps_availqty",
                "t6"."ps_supplycost" AS "ps_supplycost",
                "t6"."ps_comment" AS "ps_comment",
                "t7"."s_suppkey" AS "s_suppkey",
                "t7"."s_name" AS "s_name",
                "t7"."s_address" AS "s_address",
                "t7"."s_nationkey" AS "s_nationkey",
                "t7"."s_phone" AS "s_phone",
                "t7"."s_acctbal" AS "s_acctbal",
                "t7"."s_comment" AS "s_comment",
                "t8"."n_nationkey" AS "n_nationkey",
                "t8"."n_name" AS "n_name",
                "t8"."n_regionkey" AS "n_regionkey",
                "t8"."n_comment" AS "n_comment",
                "t9"."r_regionkey" AS "r_regionkey",
                "t9"."r_name" AS "r_name",
                "t9"."r_comment" AS "r_comment"
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
              ) AS "t7"
                ON "t7"."s_suppkey" = "t6"."ps_suppkey"
              INNER JOIN (
                SELECT
                  "t3"."N_NATIONKEY" AS "n_nationkey",
                  "t3"."N_NAME" AS "n_name",
                  "t3"."N_REGIONKEY" AS "n_regionkey",
                  "t3"."N_COMMENT" AS "n_comment"
                FROM "NATION" AS "t3"
              ) AS "t8"
                ON "t7"."s_nationkey" = "t8"."n_nationkey"
              INNER JOIN (
                SELECT
                  "t4"."R_REGIONKEY" AS "r_regionkey",
                  "t4"."R_NAME" AS "r_name",
                  "t4"."R_COMMENT" AS "r_comment"
                FROM "REGION" AS "t4"
              ) AS "t9"
                ON "t8"."n_regionkey" = "t9"."r_regionkey"
            ) AS "t15"
            WHERE
              (
                "t15"."r_name" = 'EUROPE'
              ) AND (
                "t14"."p_partkey" = "t15"."ps_partkey"
              )
          ) AS "t16"
        )
      )
  ) AS "t18"
) AS "t19"
ORDER BY
  "t19"."s_acctbal" DESC NULLS LAST,
  "t19"."n_name" ASC,
  "t19"."s_name" ASC,
  "t19"."p_partkey" ASC
LIMIT 100