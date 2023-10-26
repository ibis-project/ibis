SELECT
  t0.s_acctbal,
  t0.s_name,
  t0.n_name,
  t0.p_partkey,
  t0.p_mfgr,
  t0.s_address,
  t0.s_phone,
  t0.s_comment
FROM (
  SELECT
    t3.s_acctbal AS s_acctbal,
    t3.s_name AS s_name,
    t4.n_name AS n_name,
    t1.p_partkey AS p_partkey,
    t1.p_mfgr AS p_mfgr,
    t3.s_address AS s_address,
    t3.s_phone AS s_phone,
    t3.s_comment AS s_comment
  FROM hive.ibis_sf1.part AS t1
  JOIN hive.ibis_sf1.partsupp AS t2
    ON t1.p_partkey = t2.ps_partkey
  JOIN hive.ibis_sf1.supplier AS t3
    ON t3.s_suppkey = t2.ps_suppkey
  JOIN hive.ibis_sf1.nation AS t4
    ON t3.s_nationkey = t4.n_nationkey
  JOIN hive.ibis_sf1.region AS t5
    ON t4.n_regionkey = t5.r_regionkey
  WHERE
    t1.p_size = 15
    AND t1.p_type LIKE '%BRASS'
    AND t5.r_name = 'EUROPE'
    AND t2.ps_supplycost = (
      SELECT
        MIN(t2.ps_supplycost) AS "Min(ps_supplycost)"
      FROM hive.ibis_sf1.partsupp AS t2
      JOIN hive.ibis_sf1.supplier AS t3
        ON t3.s_suppkey = t2.ps_suppkey
      JOIN hive.ibis_sf1.nation AS t4
        ON t3.s_nationkey = t4.n_nationkey
      JOIN hive.ibis_sf1.region AS t5
        ON t4.n_regionkey = t5.r_regionkey
      WHERE
        t5.r_name = 'EUROPE' AND t1.p_partkey = t2.ps_partkey
    )
) AS t0
ORDER BY
  t0.s_acctbal DESC,
  t0.n_name ASC,
  t0.s_name ASC,
  t0.p_partkey ASC
LIMIT 100