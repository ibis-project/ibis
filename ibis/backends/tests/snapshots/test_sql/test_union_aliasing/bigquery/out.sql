WITH t0 AS (
  SELECT
    t7.`field_of_study`,
    IF(pos = pos_2, __pivoted__, NULL) AS __pivoted__
  FROM humanities AS t7, UNNEST(GENERATE_ARRAY(
    0,
    GREATEST(
      ARRAY_LENGTH(
        [STRUCT('1970-71' AS years, t7.`1970-71` AS degrees), STRUCT('1975-76' AS years, t7.`1975-76` AS degrees), STRUCT('1980-81' AS years, t7.`1980-81` AS degrees), STRUCT('1985-86' AS years, t7.`1985-86` AS degrees), STRUCT('1990-91' AS years, t7.`1990-91` AS degrees), STRUCT('1995-96' AS years, t7.`1995-96` AS degrees), STRUCT('2000-01' AS years, t7.`2000-01` AS degrees), STRUCT('2005-06' AS years, t7.`2005-06` AS degrees), STRUCT('2010-11' AS years, t7.`2010-11` AS degrees), STRUCT('2011-12' AS years, t7.`2011-12` AS degrees), STRUCT('2012-13' AS years, t7.`2012-13` AS degrees), STRUCT('2013-14' AS years, t7.`2013-14` AS degrees), STRUCT('2014-15' AS years, t7.`2014-15` AS degrees), STRUCT('2015-16' AS years, t7.`2015-16` AS degrees), STRUCT('2016-17' AS years, t7.`2016-17` AS degrees), STRUCT('2017-18' AS years, t7.`2017-18` AS degrees), STRUCT('2018-19' AS years, t7.`2018-19` AS degrees), STRUCT('2019-20' AS years, t7.`2019-20` AS degrees)]
      )
    ) - 1
  )) AS pos
  CROSS JOIN UNNEST([STRUCT('1970-71' AS years, t7.`1970-71` AS degrees), STRUCT('1975-76' AS years, t7.`1975-76` AS degrees), STRUCT('1980-81' AS years, t7.`1980-81` AS degrees), STRUCT('1985-86' AS years, t7.`1985-86` AS degrees), STRUCT('1990-91' AS years, t7.`1990-91` AS degrees), STRUCT('1995-96' AS years, t7.`1995-96` AS degrees), STRUCT('2000-01' AS years, t7.`2000-01` AS degrees), STRUCT('2005-06' AS years, t7.`2005-06` AS degrees), STRUCT('2010-11' AS years, t7.`2010-11` AS degrees), STRUCT('2011-12' AS years, t7.`2011-12` AS degrees), STRUCT('2012-13' AS years, t7.`2012-13` AS degrees), STRUCT('2013-14' AS years, t7.`2013-14` AS degrees), STRUCT('2014-15' AS years, t7.`2014-15` AS degrees), STRUCT('2015-16' AS years, t7.`2015-16` AS degrees), STRUCT('2016-17' AS years, t7.`2016-17` AS degrees), STRUCT('2017-18' AS years, t7.`2017-18` AS degrees), STRUCT('2018-19' AS years, t7.`2018-19` AS degrees), STRUCT('2019-20' AS years, t7.`2019-20` AS degrees)]) AS __pivoted__ WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(
          [STRUCT('1970-71' AS years, t7.`1970-71` AS degrees), STRUCT('1975-76' AS years, t7.`1975-76` AS degrees), STRUCT('1980-81' AS years, t7.`1980-81` AS degrees), STRUCT('1985-86' AS years, t7.`1985-86` AS degrees), STRUCT('1990-91' AS years, t7.`1990-91` AS degrees), STRUCT('1995-96' AS years, t7.`1995-96` AS degrees), STRUCT('2000-01' AS years, t7.`2000-01` AS degrees), STRUCT('2005-06' AS years, t7.`2005-06` AS degrees), STRUCT('2010-11' AS years, t7.`2010-11` AS degrees), STRUCT('2011-12' AS years, t7.`2011-12` AS degrees), STRUCT('2012-13' AS years, t7.`2012-13` AS degrees), STRUCT('2013-14' AS years, t7.`2013-14` AS degrees), STRUCT('2014-15' AS years, t7.`2014-15` AS degrees), STRUCT('2015-16' AS years, t7.`2015-16` AS degrees), STRUCT('2016-17' AS years, t7.`2016-17` AS degrees), STRUCT('2017-18' AS years, t7.`2017-18` AS degrees), STRUCT('2018-19' AS years, t7.`2018-19` AS degrees), STRUCT('2019-20' AS years, t7.`2019-20` AS degrees)]
        ) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(
          [STRUCT('1970-71' AS years, t7.`1970-71` AS degrees), STRUCT('1975-76' AS years, t7.`1975-76` AS degrees), STRUCT('1980-81' AS years, t7.`1980-81` AS degrees), STRUCT('1985-86' AS years, t7.`1985-86` AS degrees), STRUCT('1990-91' AS years, t7.`1990-91` AS degrees), STRUCT('1995-96' AS years, t7.`1995-96` AS degrees), STRUCT('2000-01' AS years, t7.`2000-01` AS degrees), STRUCT('2005-06' AS years, t7.`2005-06` AS degrees), STRUCT('2010-11' AS years, t7.`2010-11` AS degrees), STRUCT('2011-12' AS years, t7.`2011-12` AS degrees), STRUCT('2012-13' AS years, t7.`2012-13` AS degrees), STRUCT('2013-14' AS years, t7.`2013-14` AS degrees), STRUCT('2014-15' AS years, t7.`2014-15` AS degrees), STRUCT('2015-16' AS years, t7.`2015-16` AS degrees), STRUCT('2016-17' AS years, t7.`2016-17` AS degrees), STRUCT('2017-18' AS years, t7.`2017-18` AS degrees), STRUCT('2018-19' AS years, t7.`2018-19` AS degrees), STRUCT('2019-20' AS years, t7.`2019-20` AS degrees)]
        ) - 1
      )
    )
), t1 AS (
  SELECT
    t0.`field_of_study`,
    t0.`__pivoted__`.`years` AS `years`,
    t0.`__pivoted__`.`degrees` AS `degrees`
  FROM t0
), t2 AS (
  SELECT
    t1.*,
    first_value(t1.`degrees`) OVER (PARTITION BY t1.`field_of_study` ORDER BY t1.`years` ASC) AS `earliest_degrees`,
    last_value(t1.`degrees`) OVER (PARTITION BY t1.`field_of_study` ORDER BY t1.`years` ASC) AS `latest_degrees`
  FROM t1
), t3 AS (
  SELECT
    t2.*,
    t2.`latest_degrees` - t2.`earliest_degrees` AS `diff`
  FROM t2
), t4 AS (
  SELECT
    t3.`field_of_study`,
    ANY_VALUE(t3.`diff`) AS `diff`
  FROM t3
  GROUP BY
    1
), t5 AS (
  SELECT
    t4.*
  FROM t4
  WHERE
    t4.`diff` < 0
)
SELECT
  t6.`field_of_study`,
  t6.`diff`
FROM (
  WITH t0 AS (
    SELECT
      t7.`field_of_study`,
      IF(pos = pos_2, __pivoted__, NULL) AS __pivoted__
    FROM humanities AS t7, UNNEST(GENERATE_ARRAY(
      0,
      GREATEST(
        ARRAY_LENGTH(
          [STRUCT('1970-71' AS years, t7.`1970-71` AS degrees), STRUCT('1975-76' AS years, t7.`1975-76` AS degrees), STRUCT('1980-81' AS years, t7.`1980-81` AS degrees), STRUCT('1985-86' AS years, t7.`1985-86` AS degrees), STRUCT('1990-91' AS years, t7.`1990-91` AS degrees), STRUCT('1995-96' AS years, t7.`1995-96` AS degrees), STRUCT('2000-01' AS years, t7.`2000-01` AS degrees), STRUCT('2005-06' AS years, t7.`2005-06` AS degrees), STRUCT('2010-11' AS years, t7.`2010-11` AS degrees), STRUCT('2011-12' AS years, t7.`2011-12` AS degrees), STRUCT('2012-13' AS years, t7.`2012-13` AS degrees), STRUCT('2013-14' AS years, t7.`2013-14` AS degrees), STRUCT('2014-15' AS years, t7.`2014-15` AS degrees), STRUCT('2015-16' AS years, t7.`2015-16` AS degrees), STRUCT('2016-17' AS years, t7.`2016-17` AS degrees), STRUCT('2017-18' AS years, t7.`2017-18` AS degrees), STRUCT('2018-19' AS years, t7.`2018-19` AS degrees), STRUCT('2019-20' AS years, t7.`2019-20` AS degrees)]
        )
      ) - 1
    )) AS pos
    CROSS JOIN UNNEST([STRUCT('1970-71' AS years, t7.`1970-71` AS degrees), STRUCT('1975-76' AS years, t7.`1975-76` AS degrees), STRUCT('1980-81' AS years, t7.`1980-81` AS degrees), STRUCT('1985-86' AS years, t7.`1985-86` AS degrees), STRUCT('1990-91' AS years, t7.`1990-91` AS degrees), STRUCT('1995-96' AS years, t7.`1995-96` AS degrees), STRUCT('2000-01' AS years, t7.`2000-01` AS degrees), STRUCT('2005-06' AS years, t7.`2005-06` AS degrees), STRUCT('2010-11' AS years, t7.`2010-11` AS degrees), STRUCT('2011-12' AS years, t7.`2011-12` AS degrees), STRUCT('2012-13' AS years, t7.`2012-13` AS degrees), STRUCT('2013-14' AS years, t7.`2013-14` AS degrees), STRUCT('2014-15' AS years, t7.`2014-15` AS degrees), STRUCT('2015-16' AS years, t7.`2015-16` AS degrees), STRUCT('2016-17' AS years, t7.`2016-17` AS degrees), STRUCT('2017-18' AS years, t7.`2017-18` AS degrees), STRUCT('2018-19' AS years, t7.`2018-19` AS degrees), STRUCT('2019-20' AS years, t7.`2019-20` AS degrees)]) AS __pivoted__ WITH OFFSET AS pos_2
    WHERE
      pos = pos_2
      OR (
        pos > (
          ARRAY_LENGTH(
            [STRUCT('1970-71' AS years, t7.`1970-71` AS degrees), STRUCT('1975-76' AS years, t7.`1975-76` AS degrees), STRUCT('1980-81' AS years, t7.`1980-81` AS degrees), STRUCT('1985-86' AS years, t7.`1985-86` AS degrees), STRUCT('1990-91' AS years, t7.`1990-91` AS degrees), STRUCT('1995-96' AS years, t7.`1995-96` AS degrees), STRUCT('2000-01' AS years, t7.`2000-01` AS degrees), STRUCT('2005-06' AS years, t7.`2005-06` AS degrees), STRUCT('2010-11' AS years, t7.`2010-11` AS degrees), STRUCT('2011-12' AS years, t7.`2011-12` AS degrees), STRUCT('2012-13' AS years, t7.`2012-13` AS degrees), STRUCT('2013-14' AS years, t7.`2013-14` AS degrees), STRUCT('2014-15' AS years, t7.`2014-15` AS degrees), STRUCT('2015-16' AS years, t7.`2015-16` AS degrees), STRUCT('2016-17' AS years, t7.`2016-17` AS degrees), STRUCT('2017-18' AS years, t7.`2017-18` AS degrees), STRUCT('2018-19' AS years, t7.`2018-19` AS degrees), STRUCT('2019-20' AS years, t7.`2019-20` AS degrees)]
          ) - 1
        )
        AND pos_2 = (
          ARRAY_LENGTH(
            [STRUCT('1970-71' AS years, t7.`1970-71` AS degrees), STRUCT('1975-76' AS years, t7.`1975-76` AS degrees), STRUCT('1980-81' AS years, t7.`1980-81` AS degrees), STRUCT('1985-86' AS years, t7.`1985-86` AS degrees), STRUCT('1990-91' AS years, t7.`1990-91` AS degrees), STRUCT('1995-96' AS years, t7.`1995-96` AS degrees), STRUCT('2000-01' AS years, t7.`2000-01` AS degrees), STRUCT('2005-06' AS years, t7.`2005-06` AS degrees), STRUCT('2010-11' AS years, t7.`2010-11` AS degrees), STRUCT('2011-12' AS years, t7.`2011-12` AS degrees), STRUCT('2012-13' AS years, t7.`2012-13` AS degrees), STRUCT('2013-14' AS years, t7.`2013-14` AS degrees), STRUCT('2014-15' AS years, t7.`2014-15` AS degrees), STRUCT('2015-16' AS years, t7.`2015-16` AS degrees), STRUCT('2016-17' AS years, t7.`2016-17` AS degrees), STRUCT('2017-18' AS years, t7.`2017-18` AS degrees), STRUCT('2018-19' AS years, t7.`2018-19` AS degrees), STRUCT('2019-20' AS years, t7.`2019-20` AS degrees)]
          ) - 1
        )
      )
  ), t1 AS (
    SELECT
      t0.`field_of_study`,
      t0.`__pivoted__`.`years` AS `years`,
      t0.`__pivoted__`.`degrees` AS `degrees`
    FROM t0
  ), t2 AS (
    SELECT
      t1.*,
      first_value(t1.`degrees`) OVER (PARTITION BY t1.`field_of_study` ORDER BY t1.`years` ASC) AS `earliest_degrees`,
      last_value(t1.`degrees`) OVER (PARTITION BY t1.`field_of_study` ORDER BY t1.`years` ASC) AS `latest_degrees`
    FROM t1
  ), t3 AS (
    SELECT
      t2.*,
      t2.`latest_degrees` - t2.`earliest_degrees` AS `diff`
    FROM t2
  ), t4 AS (
    SELECT
      t3.`field_of_study`,
      ANY_VALUE(t3.`diff`) AS `diff`
    FROM t3
    GROUP BY
      1
  ), t5 AS (
    SELECT
      t4.*
    FROM t4
    WHERE
      t4.`diff` < 0
  ), t7 AS (
    SELECT
      t5.*
    FROM t5
    ORDER BY
      t5.`diff` ASC
  ), t8 AS (
    SELECT
      t4.*
    FROM t4
    ORDER BY
      t4.`diff` DESC
  ), t9 AS (
    SELECT
      t5.*
    FROM t5
    ORDER BY
      t5.`diff` ASC
    LIMIT 10
  ), t10 AS (
    SELECT
      t4.*
    FROM t4
    ORDER BY
      t4.`diff` DESC
    LIMIT 10
  )
  SELECT
    *
  FROM t10
  UNION ALL
  SELECT
    *
  FROM t9
) AS t6