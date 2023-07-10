SELECT
  t3.r_name AS r_name,
  t4.n_name AS n_name
FROM (
  SELECT
    t0.r_regionkey AS r_regionkey,
    t0.r_name AS r_name,
    t0.r_comment AS r_comment,
    t1.n_nationkey AS n_nationkey,
    t1.n_name AS n_name,
    t1.n_regionkey AS n_regionkey,
    t1.n_comment AS n_comment
  FROM tpch_region AS t0
  INNER JOIN tpch_nation AS t1
    ON t0.r_regionkey = t1.n_regionkey
) AS t3
INNER JOIN (
  SELECT
    t0.r_regionkey AS r_regionkey,
    t0.r_name AS r_name,
    t0.r_comment AS r_comment,
    t1.n_nationkey AS n_nationkey,
    t1.n_name AS n_name,
    t1.n_regionkey AS n_regionkey,
    t1.n_comment AS n_comment
  FROM tpch_region AS t0
  INNER JOIN tpch_nation AS t1
    ON t0.r_regionkey = t1.n_regionkey
) AS t4
  ON t3.r_regionkey = t4.r_regionkey