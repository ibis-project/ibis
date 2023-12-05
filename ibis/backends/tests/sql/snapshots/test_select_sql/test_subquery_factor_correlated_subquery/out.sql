WITH t0 AS (
  SELECT t3.*, t1.`r_name` AS `region`, t4.`o_totalprice` AS `amount`,
         CAST(t4.`o_orderdate` AS timestamp) AS `odate`
  FROM tpch_region t1
    INNER JOIN tpch_nation t2
      ON t1.`r_regionkey` = t2.`n_regionkey`
    INNER JOIN tpch_customer t3
      ON t3.`c_nationkey` = t2.`n_nationkey`
    INNER JOIN tpch_orders t4
      ON t4.`o_custkey` = t3.`c_custkey`
)
SELECT t0.*
FROM t0
WHERE t0.`amount` > (
  SELECT avg(t1.`amount`) AS `Mean(amount)`
  FROM t0 t1
  WHERE t1.`region` = t0.`region`
)
LIMIT 10