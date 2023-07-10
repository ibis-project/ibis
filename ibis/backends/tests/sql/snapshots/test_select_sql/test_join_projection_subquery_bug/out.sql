SELECT
  t1.c_custkey AS c_custkey,
  t1.c_name AS c_name,
  t1.c_address AS c_address,
  t1.c_nationkey AS c_nationkey,
  t1.c_phone AS c_phone,
  t1.c_acctbal AS c_acctbal,
  t1.c_mktsegment AS c_mktsegment,
  t1.c_comment AS c_comment,
  t4.n_nationkey AS n_nationkey,
  t4.nation AS nation,
  t4.region AS region
FROM (
  SELECT
    t0.n_nationkey AS n_nationkey,
    t0.n_name AS nation,
    t2.r_name AS region
  FROM tpch_nation AS t0
  INNER JOIN tpch_region AS t2
    ON t0.n_regionkey = t2.r_regionkey
) AS t4
INNER JOIN tpch_customer AS t1
  ON t4.n_nationkey = t1.c_nationkey