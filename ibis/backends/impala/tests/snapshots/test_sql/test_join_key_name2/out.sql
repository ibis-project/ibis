WITH `t9` AS (
  SELECT
    EXTRACT(year FROM `t8`.`odate`) AS `year`,
    COUNT(*) AS `CountStar()`
  FROM (
    SELECT
      `t6`.`c_custkey`,
      `t6`.`c_name`,
      `t6`.`c_address`,
      `t6`.`c_nationkey`,
      `t6`.`c_phone`,
      `t6`.`c_acctbal`,
      `t6`.`c_mktsegment`,
      `t6`.`c_comment`,
      `t4`.`r_name` AS `region`,
      `t7`.`o_totalprice`,
      CAST(`t7`.`o_orderdate` AS TIMESTAMP) AS `odate`
    FROM `tpch_region` AS `t4`
    INNER JOIN `tpch_nation` AS `t5`
      ON `t4`.`r_regionkey` = `t5`.`n_regionkey`
    INNER JOIN `tpch_customer` AS `t6`
      ON `t6`.`c_nationkey` = `t5`.`n_nationkey`
    INNER JOIN `tpch_orders` AS `t7`
      ON `t7`.`o_custkey` = `t6`.`c_custkey`
  ) AS `t8`
  GROUP BY
    1
)
SELECT
  `t11`.`year`,
  `t11`.`CountStar()` AS `pre_count`,
  `t13`.`CountStar()` AS `post_count`
FROM `t9` AS `t11`
INNER JOIN `t9` AS `t13`
  ON `t11`.`year` = `t13`.`year`