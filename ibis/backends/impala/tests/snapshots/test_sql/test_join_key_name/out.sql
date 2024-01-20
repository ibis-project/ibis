WITH `t11` AS (
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
)
SELECT
  `t15`.`year`,
  `t15`.`CountStar()` AS `pre_count`,
  `t20`.`CountStar()` AS `post_count`,
  `t20`.`CountStar()` / CAST(`t15`.`CountStar()` AS DOUBLE) AS `fraction`
FROM (
  SELECT
    EXTRACT(year FROM `t12`.`odate`) AS `year`,
    COUNT(*) AS `CountStar()`
  FROM `t11` AS `t12`
  GROUP BY
    1
) AS `t15`
INNER JOIN (
  SELECT
    EXTRACT(year FROM `t18`.`odate`) AS `year`,
    COUNT(*) AS `CountStar()`
  FROM (
    SELECT
      `t12`.`c_custkey`,
      `t12`.`c_name`,
      `t12`.`c_address`,
      `t12`.`c_nationkey`,
      `t12`.`c_phone`,
      `t12`.`c_acctbal`,
      `t12`.`c_mktsegment`,
      `t12`.`c_comment`,
      `t12`.`region`,
      `t12`.`o_totalprice`,
      `t12`.`odate`
    FROM `t11` AS `t12`
    WHERE
      `t12`.`o_totalprice` > (
        SELECT
          AVG(`t16`.`o_totalprice`) AS `Mean(o_totalprice)`
        FROM (
          SELECT
            `t13`.`c_custkey`,
            `t13`.`c_name`,
            `t13`.`c_address`,
            `t13`.`c_nationkey`,
            `t13`.`c_phone`,
            `t13`.`c_acctbal`,
            `t13`.`c_mktsegment`,
            `t13`.`c_comment`,
            `t13`.`region`,
            `t13`.`o_totalprice`,
            `t13`.`odate`
          FROM `t11` AS `t13`
          WHERE
            `t13`.`region` = `t12`.`region`
        ) AS `t16`
      )
  ) AS `t18`
  GROUP BY
    1
) AS `t20`
  ON `t15`.`year` = `t20`.`year`