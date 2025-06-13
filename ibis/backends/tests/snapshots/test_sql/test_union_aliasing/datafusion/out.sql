WITH "t5" AS (
  SELECT
    "t4"."field_of_study",
    FIRST_VALUE("t4"."diff") FILTER(WHERE
      NOT "t4"."diff" IS NULL) AS "diff"
  FROM (
    SELECT
      "t4"."years",
      "t4"."degrees",
      "t4"."earliest_degrees",
      "t4"."latest_degrees",
      "t4"."diff",
      "t4"."field_of_study"
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
          FIRST_VALUE("t2"."degrees") OVER (
            PARTITION BY "t2"."field_of_study"
            ORDER BY "t2"."years" ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
          ) AS "earliest_degrees",
          LAST_VALUE("t2"."degrees") OVER (
            PARTITION BY "t2"."field_of_study"
            ORDER BY "t2"."years" ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
          ) AS "latest_degrees"
        FROM (
          SELECT
            "t1"."field_of_study",
            "t1"."__pivoted__"['years'] AS "years",
            "t1"."__pivoted__"['degrees'] AS "degrees"
          FROM (
            SELECT
              "t0"."field_of_study",
              UNNEST(
                MAKE_ARRAY(
                  NAMED_STRUCT('years', '1970-71', 'degrees', "t0"."1970-71"),
                  NAMED_STRUCT('years', '1975-76', 'degrees', "t0"."1975-76"),
                  NAMED_STRUCT('years', '1980-81', 'degrees', "t0"."1980-81"),
                  NAMED_STRUCT('years', '1985-86', 'degrees', "t0"."1985-86"),
                  NAMED_STRUCT('years', '1990-91', 'degrees', "t0"."1990-91"),
                  NAMED_STRUCT('years', '1995-96', 'degrees', "t0"."1995-96"),
                  NAMED_STRUCT('years', '2000-01', 'degrees', "t0"."2000-01"),
                  NAMED_STRUCT('years', '2005-06', 'degrees', "t0"."2005-06"),
                  NAMED_STRUCT('years', '2010-11', 'degrees', "t0"."2010-11"),
                  NAMED_STRUCT('years', '2011-12', 'degrees', "t0"."2011-12"),
                  NAMED_STRUCT('years', '2012-13', 'degrees', "t0"."2012-13"),
                  NAMED_STRUCT('years', '2013-14', 'degrees', "t0"."2013-14"),
                  NAMED_STRUCT('years', '2014-15', 'degrees', "t0"."2014-15"),
                  NAMED_STRUCT('years', '2015-16', 'degrees', "t0"."2015-16"),
                  NAMED_STRUCT('years', '2016-17', 'degrees', "t0"."2016-17"),
                  NAMED_STRUCT('years', '2017-18', 'degrees', "t0"."2017-18"),
                  NAMED_STRUCT('years', '2018-19', 'degrees', "t0"."2018-19"),
                  NAMED_STRUCT('years', '2019-20', 'degrees', "t0"."2019-20")
                )
              ) AS "__pivoted__"
            FROM "humanities" AS "t0"
          ) AS "t1"
        ) AS "t2"
      ) AS "t3"
    ) AS "t4"
  ) AS t4
  GROUP BY
    "t4"."field_of_study"
)
SELECT
  *
FROM (
  SELECT
    *
  FROM "t5" AS "t6"
  ORDER BY
    "t6"."diff" DESC NULLS LAST
  LIMIT 10
) AS "t9"
UNION ALL
SELECT
  *
FROM (
  SELECT
    *
  FROM "t5" AS "t6"
  WHERE
    "t6"."diff" < 0
  ORDER BY
    "t6"."diff" ASC
  LIMIT 10
) AS "t10"