WITH t0 AS (
  SELECT
    t2.p_partkey AS p_partkey,
    t2.p_name AS p_name,
    t2.p_mfgr AS p_mfgr,
    t2.p_brand AS p_brand,
    t2.p_type AS p_type,
    t2.p_size AS p_size,
    t2.p_container AS p_container,
    t2.p_retailprice AS p_retailprice,
    t2.p_comment AS p_comment,
    t3.ps_partkey AS ps_partkey,
    t3.ps_suppkey AS ps_suppkey,
    t3.ps_availqty AS ps_availqty,
    t3.ps_supplycost AS ps_supplycost,
    t3.ps_comment AS ps_comment,
    t4.s_suppkey AS s_suppkey,
    t4.s_name AS s_name,
    t4.s_address AS s_address,
    t4.s_nationkey AS s_nationkey,
    t4.s_phone AS s_phone,
    t4.s_acctbal AS s_acctbal,
    t4.s_comment AS s_comment,
    t5.n_nationkey AS n_nationkey,
    t5.n_name AS n_name,
    t5.n_regionkey AS n_regionkey,
    t5.n_comment AS n_comment,
    t6.r_regionkey AS r_regionkey,
    t6.r_name AS r_name,
    t6.r_comment AS r_comment
  FROM hive.ibis_sf1.part AS t2
  JOIN hive.ibis_sf1.partsupp AS t3
    ON t2.p_partkey = t3.ps_partkey
  JOIN hive.ibis_sf1.supplier AS t4
    ON t4.s_suppkey = t3.ps_suppkey
  JOIN hive.ibis_sf1.nation AS t5
    ON t4.s_nationkey = t5.n_nationkey
  JOIN hive.ibis_sf1.region AS t6
    ON t5.n_regionkey = t6.r_regionkey
  WHERE
    t2.p_size = 15
    AND t2.p_type LIKE '%BRASS'
    AND t6.r_name = 'EUROPE'
    AND t3.ps_supplycost = (
      SELECT
        MIN(t3.ps_supplycost) AS "Min(ps_supplycost)"
      FROM hive.ibis_sf1.partsupp AS t3
      JOIN hive.ibis_sf1.supplier AS t4
        ON t4.s_suppkey = t3.ps_suppkey
      JOIN hive.ibis_sf1.nation AS t5
        ON t4.s_nationkey = t5.n_nationkey
      JOIN hive.ibis_sf1.region AS t6
        ON t5.n_regionkey = t6.r_regionkey
      WHERE
        t6.r_name = 'EUROPE' AND t2.p_partkey = t3.ps_partkey
    )
)
SELECT
  t1.s_acctbal,
  t1.s_name,
  t1.n_name,
  t1.p_partkey,
  t1.p_mfgr,
  t1.s_address,
  t1.s_phone,
  t1.s_comment
FROM (
  SELECT
    t0.s_acctbal AS s_acctbal,
    t0.s_name AS s_name,
    t0.n_name AS n_name,
    t0.p_partkey AS p_partkey,
    t0.p_mfgr AS p_mfgr,
    t0.s_address AS s_address,
    t0.s_phone AS s_phone,
    t0.s_comment AS s_comment
  FROM t0
) AS t1
ORDER BY
  t1.s_acctbal DESC,
  t1.n_name ASC,
  t1.s_name ASC,
  t1.p_partkey ASC
LIMIT 100