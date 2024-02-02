WITH "t5" AS (
  SELECT
    "t4"."field_of_study",
    FIRST("t4"."diff") AS "diff"
  FROM (
    SELECT
      "t3"."field_of_study",
      "t3"."years",
      "t3"."degrees",
      "t3"."earliest_degrees",
      "t3"."latest_degrees",
      "t3"."latest_degrees" - "t3"."earliest_degrees" AS "diff"
    FROM (
      SELECT
        "t2"."field_of_study",
        "t2"."years",
        "t2"."degrees",
        FIRST("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "earliest_degrees",
        LAST("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "latest_degrees"
      FROM (
        SELECT
          "t1"."field_of_study",
          "t1"."__pivoted__"."years" AS "years",
          "t1"."__pivoted__"."degrees" AS "degrees"
        FROM (
          SELECT
            "t0"."field_of_study",
            UNNEST(
              [{'years': '1970-71', 'degrees': "t0"."1970-71"}, {'years': '1975-76', 'degrees': "t0"."1975-76"}, {'years': '1980-81', 'degrees': "t0"."1980-81"}, {'years': '1985-86', 'degrees': "t0"."1985-86"}, {'years': '1990-91', 'degrees': "t0"."1990-91"}, {'years': '1995-96', 'degrees': "t0"."1995-96"}, {'years': '2000-01', 'degrees': "t0"."2000-01"}, {'years': '2005-06', 'degrees': "t0"."2005-06"}, {'years': '2010-11', 'degrees': "t0"."2010-11"}, {'years': '2011-12', 'degrees': "t0"."2011-12"}, {'years': '2012-13', 'degrees': "t0"."2012-13"}, {'years': '2013-14', 'degrees': "t0"."2013-14"}, {'years': '2014-15', 'degrees': "t0"."2014-15"}, {'years': '2015-16', 'degrees': "t0"."2015-16"}, {'years': '2016-17', 'degrees': "t0"."2016-17"}, {'years': '2017-18', 'degrees': "t0"."2017-18"}, {'years': '2018-19', 'degrees': "t0"."2018-19"}, {'years': '2019-20', 'degrees': "t0"."2019-20"}]
            ) AS "__pivoted__"
          FROM "humanities" AS "t0"
        ) AS "t1"
      ) AS "t2"
    ) AS "t3"
  ) AS "t4"
  GROUP BY
    1
)
SELECT
  "t11"."field_of_study",
  "t11"."diff"
FROM (
  SELECT
    "t6"."field_of_study",
    "t6"."diff"
  FROM "t5" AS "t6"
  ORDER BY
    "t6"."diff" DESC
  LIMIT 10
  UNION ALL
  SELECT
    "t6"."field_of_study",
    "t6"."diff"
  FROM "t5" AS "t6"
  WHERE
    "t6"."diff" < CAST(0 AS TINYINT)
  ORDER BY
    "t6"."diff" ASC
  LIMIT 10
) AS "t11"