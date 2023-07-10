SELECT
  t0.n_nationkey AS n_nationkey,
  t0.n_name AS n_name,
  t0.n_regionkey AS n_regionkey,
  t0.n_comment AS n_comment,
  t1.r_regionkey AS r_regionkey,
  t1.r_name AS r_name,
  t1.r_comment AS r_comment,
  t2.c_custkey AS c_custkey,
  t2.c_name AS c_name,
  t2.c_address AS c_address,
  t2.c_nationkey AS c_nationkey,
  t2.c_phone AS c_phone,
  t2.c_acctbal AS c_acctbal,
  t2.c_mktsegment AS c_mktsegment,
  t2.c_comment AS c_comment
FROM tpch_nation AS t0
INNER JOIN tpch_region AS t1
  ON t0.n_regionkey = t1.r_regionkey
INNER JOIN tpch_customer AS t2
  ON t0.n_nationkey = t2.c_nationkey