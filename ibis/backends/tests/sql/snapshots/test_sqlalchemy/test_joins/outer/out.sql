SELECT
  t0.r_regionkey,
  t0.r_name,
  t0.r_comment,
  t1.n_nationkey,
  t1.n_name,
  t1.n_regionkey,
  t1.n_comment
FROM tpch_region AS t0
FULL OUTER JOIN tpch_nation AS t1
  ON t0.r_regionkey = t1.n_regionkey