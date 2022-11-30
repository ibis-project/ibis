WITH t0 AS (
  SELECT t6.*, t1.`r_name` AS `region`, t3.`o_totalprice` AS `amount`,
         CAST(t3.`o_orderdate` AS timestamp) AS `odate`
  FROM tpch_region t1
    INNER JOIN tpch_nation t2
      ON t1.`r_regionkey` = t2.`n_regionkey`
    INNER JOIN tpch_customer t6
      ON t6.`c_nationkey` = t2.`n_nationkey`
    INNER JOIN tpch_orders t3
      ON t3.`o_custkey` = t6.`c_custkey`
)
SELECT t0.*
FROM t0
WHERE t0.`amount` > (
  SELECT avg(t4.`amount`) AS `Mean(amount)`
  FROM t0 t4
  WHERE t4.`region` = t0.`region`
)
LIMIT 10