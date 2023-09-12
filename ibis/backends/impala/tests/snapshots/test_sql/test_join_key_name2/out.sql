WITH t0 AS (
  SELECT t5.*, t3.`r_name` AS `region`, t6.`o_totalprice`,
         CAST(t6.`o_orderdate` AS timestamp) AS `odate`
  FROM `tpch_region` t3
    INNER JOIN `tpch_nation` t4
      ON t3.`r_regionkey` = t4.`n_regionkey`
    INNER JOIN `tpch_customer` t5
      ON t5.`c_nationkey` = t4.`n_nationkey`
    INNER JOIN `tpch_orders` t6
      ON t6.`o_custkey` = t5.`c_custkey`
),
t1 AS (
  SELECT extract(t0.`odate`, 'year') AS `year`, count(1) AS `CountStar()`
  FROM t0
  GROUP BY 1
)
SELECT t1.`year`, t1.`CountStar()` AS `pre_count`,
       t2.`CountStar()` AS `post_count`
FROM t1
  INNER JOIN t1 t2
    ON t1.`year` = t2.`year`