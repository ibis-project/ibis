SELECT
  t19.s_acctbal AS s_acctbal,
  t19.s_name AS s_name,
  t19.n_name AS n_name,
  t19.p_partkey AS p_partkey,
  t19.p_mfgr AS p_mfgr,
  t19.s_address AS s_address,
  t19.s_phone AS s_phone,
  t19.s_comment AS s_comment
FROM (
  SELECT
    t0.p_partkey AS p_partkey,
    t0.p_name AS p_name,
    t0.p_mfgr AS p_mfgr,
    t0.p_brand AS p_brand,
    t0.p_type AS p_type,
    t0.p_size AS p_size,
    t0.p_container AS p_container,
    t0.p_retailprice AS p_retailprice,
    t0.p_comment AS p_comment,
    t5.ps_partkey AS ps_partkey,
    t5.ps_suppkey AS ps_suppkey,
    t5.ps_availqty AS ps_availqty,
    t5.ps_supplycost AS ps_supplycost,
    t5.ps_comment AS ps_comment,
    t6.s_suppkey AS s_suppkey,
    t6.s_name AS s_name,
    t6.s_address AS s_address,
    t6.s_nationkey AS s_nationkey,
    t6.s_phone AS s_phone,
    t6.s_acctbal AS s_acctbal,
    t6.s_comment AS s_comment,
    t8.n_nationkey AS n_nationkey,
    t8.n_name AS n_name,
    t8.n_regionkey AS n_regionkey,
    t8.n_comment AS n_comment,
    t10.r_regionkey AS r_regionkey,
    t10.r_name AS r_name,
    t10.r_comment AS r_comment
  FROM part AS t0
  INNER JOIN partsupp AS t5
    ON t0.p_partkey = t5.ps_partkey
  INNER JOIN supplier AS t6
    ON t6.s_suppkey = t5.ps_suppkey
  INNER JOIN nation AS t8
    ON t6.s_nationkey = t8.n_nationkey
  INNER JOIN region AS t10
    ON t8.n_regionkey = t10.r_regionkey
) AS t19
WHERE
  t19.p_size = CAST(15 AS TINYINT)
  AND t19.p_type LIKE '%BRASS'
  AND t19.r_name = 'EUROPE'
  AND t19.ps_supplycost = (
    SELECT
      MIN(t21.ps_supplycost) AS "Min(ps_supplycost)"
    FROM (
      SELECT
        t20.ps_partkey AS ps_partkey,
        t20.ps_suppkey AS ps_suppkey,
        t20.ps_availqty AS ps_availqty,
        t20.ps_supplycost AS ps_supplycost,
        t20.ps_comment AS ps_comment,
        t20.s_suppkey AS s_suppkey,
        t20.s_name AS s_name,
        t20.s_address AS s_address,
        t20.s_nationkey AS s_nationkey,
        t20.s_phone AS s_phone,
        t20.s_acctbal AS s_acctbal,
        t20.s_comment AS s_comment,
        t20.n_nationkey AS n_nationkey,
        t20.n_name AS n_name,
        t20.n_regionkey AS n_regionkey,
        t20.n_comment AS n_comment,
        t20.r_regionkey AS r_regionkey,
        t20.r_name AS r_name,
        t20.r_comment AS r_comment
      FROM (
        SELECT
          t1.ps_partkey AS ps_partkey,
          t1.ps_suppkey AS ps_suppkey,
          t1.ps_availqty AS ps_availqty,
          t1.ps_supplycost AS ps_supplycost,
          t1.ps_comment AS ps_comment,
          t7.s_suppkey AS s_suppkey,
          t7.s_name AS s_name,
          t7.s_address AS s_address,
          t7.s_nationkey AS s_nationkey,
          t7.s_phone AS s_phone,
          t7.s_acctbal AS s_acctbal,
          t7.s_comment AS s_comment,
          t9.n_nationkey AS n_nationkey,
          t9.n_name AS n_name,
          t9.n_regionkey AS n_regionkey,
          t9.n_comment AS n_comment,
          t11.r_regionkey AS r_regionkey,
          t11.r_name AS r_name,
          t11.r_comment AS r_comment
        FROM partsupp AS t1
        INNER JOIN supplier AS t7
          ON t7.s_suppkey = t1.ps_suppkey
        INNER JOIN nation AS t9
          ON t7.s_nationkey = t9.n_nationkey
        INNER JOIN region AS t11
          ON t9.n_regionkey = t11.r_regionkey
      ) AS t20
      WHERE
        t20.r_name = 'EUROPE' AND t19.p_partkey = t20.ps_partkey
    ) AS t21
  )
ORDER BY
  t19.s_acctbal DESC,
  t19.n_name ASC,
  t19.s_name ASC,
  t19.p_partkey ASC
LIMIT 100