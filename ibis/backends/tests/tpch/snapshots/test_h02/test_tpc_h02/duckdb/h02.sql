SELECT
  t21.s_acctbal,
  t21.s_name,
  t21.n_name,
  t21.p_partkey,
  t21.p_mfgr,
  t21.s_address,
  t21.s_phone,
  t21.s_comment
FROM (
  SELECT
    t5.p_partkey,
    t5.p_name,
    t5.p_mfgr,
    t5.p_brand,
    t5.p_type,
    t5.p_size,
    t5.p_container,
    t5.p_retailprice,
    t5.p_comment,
    t6.ps_partkey,
    t6.ps_suppkey,
    t6.ps_availqty,
    t6.ps_supplycost,
    t6.ps_comment,
    t8.s_suppkey,
    t8.s_name,
    t8.s_address,
    t8.s_nationkey,
    t8.s_phone,
    t8.s_acctbal,
    t8.s_comment,
    t10.n_nationkey,
    t10.n_name,
    t10.n_regionkey,
    t10.n_comment,
    t12.r_regionkey,
    t12.r_name,
    t12.r_comment
  FROM part AS t5
  INNER JOIN partsupp AS t6
    ON t5.p_partkey = t6.ps_partkey
  INNER JOIN supplier AS t8
    ON t8.s_suppkey = t6.ps_suppkey
  INNER JOIN nation AS t10
    ON t8.s_nationkey = t10.n_nationkey
  INNER JOIN region AS t12
    ON t10.n_regionkey = t12.r_regionkey
) AS t21
WHERE
  t21.p_size = CAST(15 AS TINYINT)
  AND t21.p_type LIKE '%BRASS'
  AND t21.r_name = 'EUROPE'
  AND t21.ps_supplycost = (
    SELECT
      MIN(t23.ps_supplycost) AS "Min(ps_supplycost)"
    FROM (
      SELECT
        t22.ps_partkey,
        t22.ps_suppkey,
        t22.ps_availqty,
        t22.ps_supplycost,
        t22.ps_comment,
        t22.s_suppkey,
        t22.s_name,
        t22.s_address,
        t22.s_nationkey,
        t22.s_phone,
        t22.s_acctbal,
        t22.s_comment,
        t22.n_nationkey,
        t22.n_name,
        t22.n_regionkey,
        t22.n_comment,
        t22.r_regionkey,
        t22.r_name,
        t22.r_comment
      FROM (
        SELECT
          t7.ps_partkey,
          t7.ps_suppkey,
          t7.ps_availqty,
          t7.ps_supplycost,
          t7.ps_comment,
          t9.s_suppkey,
          t9.s_name,
          t9.s_address,
          t9.s_nationkey,
          t9.s_phone,
          t9.s_acctbal,
          t9.s_comment,
          t11.n_nationkey,
          t11.n_name,
          t11.n_regionkey,
          t11.n_comment,
          t13.r_regionkey,
          t13.r_name,
          t13.r_comment
        FROM partsupp AS t7
        INNER JOIN supplier AS t9
          ON t9.s_suppkey = t7.ps_suppkey
        INNER JOIN nation AS t11
          ON t9.s_nationkey = t11.n_nationkey
        INNER JOIN region AS t13
          ON t11.n_regionkey = t13.r_regionkey
      ) AS t22
      WHERE
        t22.r_name = 'EUROPE' AND t21.p_partkey = t22.ps_partkey
    ) AS t23
  )
ORDER BY
  t21.s_acctbal DESC,
  t21.n_name ASC,
  t21.s_name ASC,
  t21.p_partkey ASC
LIMIT 100