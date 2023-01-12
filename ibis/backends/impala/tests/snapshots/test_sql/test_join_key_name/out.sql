WITH t0 AS (
  SELECT t5.*, t3.`r_name` AS `region`, t6.`o_totalprice`,
         CAST(t6.`o_orderdate` AS timestamp) AS `odate`
  FROM tpch_region t3
    INNER JOIN tpch_nation t4
      ON t3.`r_regionkey` = t4.`n_regionkey`
    INNER JOIN tpch_customer t5
      ON t5.`c_nationkey` = t4.`n_nationkey`
    INNER JOIN tpch_orders t6
      ON t6.`o_custkey` = t5.`c_custkey`
)
SELECT t1.`year`, t1.`count` AS `pre_count`, t2.`count` AS `post_count`,
       t2.`count` / CAST(t1.`count` AS double) AS `fraction`
FROM (
  SELECT extract(t0.`odate`, 'year') AS `year`, count(1) AS `count`
  FROM t0
  GROUP BY 1
) t1
  INNER JOIN (
    SELECT extract(t0.`odate`, 'year') AS `year`, count(1) AS `count`
    FROM t0
    WHERE t0.`o_totalprice` > (
      SELECT avg(t4.`o_totalprice`) AS `mean`
      FROM t0 t4
      WHERE t4.`region` = t0.`region`
    )
    GROUP BY 1
  ) t2
    ON t1.`year` = t2.`year`