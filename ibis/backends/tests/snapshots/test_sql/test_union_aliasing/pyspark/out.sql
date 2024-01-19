WITH `t5` AS (
  SELECT
    `t4`.`field_of_study`,
    FIRST(`t4`.`diff`, TRUE) AS `diff`
  FROM (
    SELECT
      `t3`.`field_of_study`,
      `t3`.`years`,
      `t3`.`degrees`,
      `t3`.`earliest_degrees`,
      `t3`.`latest_degrees`,
      `t3`.`latest_degrees` - `t3`.`earliest_degrees` AS `diff`
    FROM (
      SELECT
        `t2`.`field_of_study`,
        `t2`.`years`,
        `t2`.`degrees`,
        FIRST(`t2`.`degrees`, TRUE) OVER (PARTITION BY `t2`.`field_of_study` ORDER BY `t2`.`years` ASC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS `earliest_degrees`,
        LAST(`t2`.`degrees`, TRUE) OVER (PARTITION BY `t2`.`field_of_study` ORDER BY `t2`.`years` ASC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS `latest_degrees`
      FROM (
        SELECT
          `t1`.`field_of_study`,
          `t1`.`__pivoted__`.`years` AS `years`,
          `t1`.`__pivoted__`.`degrees` AS `degrees`
        FROM (
          SELECT
            `t0`.`field_of_study`,
            EXPLODE(
              ARRAY(
                STRUCT('1970-71' AS `years`, `t0`.`1970-71` AS `degrees`),
                STRUCT('1975-76' AS `years`, `t0`.`1975-76` AS `degrees`),
                STRUCT('1980-81' AS `years`, `t0`.`1980-81` AS `degrees`),
                STRUCT('1985-86' AS `years`, `t0`.`1985-86` AS `degrees`),
                STRUCT('1990-91' AS `years`, `t0`.`1990-91` AS `degrees`),
                STRUCT('1995-96' AS `years`, `t0`.`1995-96` AS `degrees`),
                STRUCT('2000-01' AS `years`, `t0`.`2000-01` AS `degrees`),
                STRUCT('2005-06' AS `years`, `t0`.`2005-06` AS `degrees`),
                STRUCT('2010-11' AS `years`, `t0`.`2010-11` AS `degrees`),
                STRUCT('2011-12' AS `years`, `t0`.`2011-12` AS `degrees`),
                STRUCT('2012-13' AS `years`, `t0`.`2012-13` AS `degrees`),
                STRUCT('2013-14' AS `years`, `t0`.`2013-14` AS `degrees`),
                STRUCT('2014-15' AS `years`, `t0`.`2014-15` AS `degrees`),
                STRUCT('2015-16' AS `years`, `t0`.`2015-16` AS `degrees`),
                STRUCT('2016-17' AS `years`, `t0`.`2016-17` AS `degrees`),
                STRUCT('2017-18' AS `years`, `t0`.`2017-18` AS `degrees`),
                STRUCT('2018-19' AS `years`, `t0`.`2018-19` AS `degrees`),
                STRUCT('2019-20' AS `years`, `t0`.`2019-20` AS `degrees`)
              )
            ) AS `__pivoted__`
          FROM `humanities` AS `t0`
        ) AS `t1`
      ) AS `t2`
    ) AS `t3`
  ) AS `t4`
  GROUP BY
    1
)
SELECT
  `t11`.`field_of_study`,
  `t11`.`diff`
FROM (
  SELECT
    `t6`.`field_of_study`,
    `t6`.`diff`
  FROM `t5` AS `t6`
  ORDER BY
    `t6`.`diff` DESC
  LIMIT 10
  UNION ALL
  SELECT
    `t6`.`field_of_study`,
    `t6`.`diff`
  FROM `t5` AS `t6`
  WHERE
    `t6`.`diff` < 0
  ORDER BY
    `t6`.`diff` ASC NULLS LAST
  LIMIT 10
) AS `t11`