SELECT
  t0.n_nationkey,
  t0.n_name,
  t0.n_regionkey,
  t0.n_comment,
  t1.r_regionkey,
  t1.r_name,
  t1.r_comment,
  t2.c_custkey,
  t2.c_name,
  t2.c_address,
  t2.c_nationkey,
  t2.c_phone,
  t2.c_acctbal,
  t2.c_mktsegment,
  t2.c_comment
FROM tpch_nation AS t0
JOIN tpch_region AS t1
  ON t0.n_regionkey = t1.r_regionkey
JOIN tpch_customer AS t2
  ON t0.n_nationkey = t2.c_nationkey