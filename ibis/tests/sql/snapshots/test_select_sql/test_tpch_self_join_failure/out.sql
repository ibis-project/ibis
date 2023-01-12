WITH t0 AS (
  SELECT t3.`r_name` AS `region`, t4.`n_name` AS `nation`,
         t6.`o_totalprice` AS `amount`,
         CAST(t6.`o_orderdate` AS timestamp) AS `odate`
  FROM tpch_region t3
    INNER JOIN tpch_nation t4
      ON t3.`r_regionkey` = t4.`n_regionkey`
    INNER JOIN tpch_customer t5
      ON t5.`c_nationkey` = t4.`n_nationkey`
    INNER JOIN tpch_orders t6
      ON t6.`o_custkey` = t5.`c_custkey`
),
t1 AS (
  SELECT t0.`region`, extract(t0.`odate`, 'year') AS `year`,
         CAST(sum(t0.`amount`) AS double) AS `total`
  FROM t0
  GROUP BY 1, 2
)
SELECT t1.`region`, t1.`year`, t1.`total` - t2.`total` AS `yoy_change`
FROM t1
  INNER JOIN t1 t2
    ON t1.`year` = (t2.`year` - 1)