WITH `t4` AS (
  SELECT
    `t3`.`field_of_study`,
    ANY_VALUE(`t3`.`diff`) AS `diff`
  FROM (
    SELECT
      `t2`.`field_of_study`,
      `t2`.`years`,
      `t2`.`degrees`,
      `t2`.`earliest_degrees`,
      `t2`.`latest_degrees`,
      `t2`.`latest_degrees` - `t2`.`earliest_degrees` AS `diff`
    FROM (
      SELECT
        `t1`.`field_of_study`,
        `t1`.`years`,
        `t1`.`degrees`,
        FIRST_VALUE(`t1`.`degrees`) OVER (
          PARTITION BY `t1`.`field_of_study`
          ORDER BY `t1`.`years` ASC
          ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS `earliest_degrees`,
        LAST_VALUE(`t1`.`degrees`) OVER (
          PARTITION BY `t1`.`field_of_study`
          ORDER BY `t1`.`years` ASC
          ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS `latest_degrees`
      FROM (
        SELECT
          `field_of_study`,
          `years`,
          `degrees`
        FROM (
          SELECT
            `t0`.*
            EXCEPT (`1970-71`, `1975-76`, `1980-81`, `1985-86`, `1990-91`, `1995-96`, `2000-01`, `2005-06`, `2010-11`, `2011-12`, `2012-13`, `2013-14`, `2014-15`, `2015-16`, `2016-17`, `2017-18`, `2018-19`, `2019-20`),
            CAST(`t0`.`1970-71` AS INT64) AS `1970-71`,
            CAST(`t0`.`1975-76` AS INT64) AS `1975-76`,
            CAST(`t0`.`1980-81` AS INT64) AS `1980-81`,
            CAST(`t0`.`1985-86` AS INT64) AS `1985-86`,
            CAST(`t0`.`1990-91` AS INT64) AS `1990-91`,
            CAST(`t0`.`1995-96` AS INT64) AS `1995-96`,
            CAST(`t0`.`2000-01` AS INT64) AS `2000-01`,
            CAST(`t0`.`2005-06` AS INT64) AS `2005-06`,
            CAST(`t0`.`2010-11` AS INT64) AS `2010-11`,
            CAST(`t0`.`2011-12` AS INT64) AS `2011-12`,
            CAST(`t0`.`2012-13` AS INT64) AS `2012-13`,
            CAST(`t0`.`2013-14` AS INT64) AS `2013-14`,
            CAST(`t0`.`2014-15` AS INT64) AS `2014-15`,
            CAST(`t0`.`2015-16` AS INT64) AS `2015-16`,
            CAST(`t0`.`2016-17` AS INT64) AS `2016-17`,
            CAST(`t0`.`2017-18` AS INT64) AS `2017-18`,
            CAST(`t0`.`2018-19` AS INT64) AS `2018-19`,
            CAST(`t0`.`2019-20` AS INT64) AS `2019-20`
          FROM `humanities` AS `t0`
        )
        UNPIVOT INCLUDE NULLS (`degrees` FOR 
          `years` IN (
            `1970-71` AS '1970-71',
            `1975-76` AS '1975-76',
            `1980-81` AS '1980-81',
            `1985-86` AS '1985-86',
            `1990-91` AS '1990-91',
            `1995-96` AS '1995-96',
            `2000-01` AS '2000-01',
            `2005-06` AS '2005-06',
            `2010-11` AS '2010-11',
            `2011-12` AS '2011-12',
            `2012-13` AS '2012-13',
            `2013-14` AS '2013-14',
            `2014-15` AS '2014-15',
            `2015-16` AS '2015-16',
            `2016-17` AS '2016-17',
            `2017-18` AS '2017-18',
            `2018-19` AS '2018-19',
            `2019-20` AS '2019-20'
          )
        )
      ) AS `t1`
    ) AS `t2`
  ) AS `t3`
  GROUP BY
    1
)
SELECT
  *
FROM (
  SELECT
    *
  FROM `t4` AS `t5`
  ORDER BY
    `t5`.`diff` DESC
  LIMIT 10
) AS `t8`
UNION ALL
SELECT
  *
FROM (
  SELECT
    *
  FROM `t4` AS `t5`
  WHERE
    `t5`.`diff` < 0
  ORDER BY
    `t5`.`diff` ASC NULLS LAST
  LIMIT 10
) AS `t9`