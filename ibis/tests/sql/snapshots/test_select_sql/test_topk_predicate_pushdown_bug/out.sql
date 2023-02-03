WITH t0 AS (
  SELECT t2.*, t3.`n_name`, t4.`r_name`
  FROM tpch_customer t2
    INNER JOIN tpch_nation t3
      ON t2.`c_nationkey` = t3.`n_nationkey`
    INNER JOIN tpch_region t4
      ON t3.`n_regionkey` = t4.`r_regionkey`
)
SELECT *
FROM t0
  LEFT SEMI JOIN (
    SELECT t2.*
    FROM (
      SELECT t0.`n_name`, sum(t0.`c_acctbal`) AS `Sum(c_acctbal)`
      FROM t0
      GROUP BY 1
    ) t2
    ORDER BY t2.`Sum(c_acctbal)` DESC
    LIMIT 10
  ) t1
    ON t0.`n_name` = t1.`n_name`