WITH "t5" AS (
  SELECT
    "t4"."field_of_study",
    any("t4"."diff") AS "diff"
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
        any("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "earliest_degrees",
        anyLast("t2"."degrees") OVER (PARTITION BY "t2"."field_of_study" ORDER BY "t2"."years" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS "latest_degrees"
      FROM (
        SELECT
          "t1"."field_of_study",
          CAST("t1"."__pivoted__".1 AS Nullable(String)) AS "years",
          CAST("t1"."__pivoted__".2 AS Nullable(Int64)) AS "degrees"
        FROM (
          SELECT
            "t0"."field_of_study",
            arrayJoin(
              [CAST(tuple('1970-71', "t0"."1970-71") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1975-76', "t0"."1975-76") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1980-81', "t0"."1980-81") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1985-86', "t0"."1985-86") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1990-91', "t0"."1990-91") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('1995-96', "t0"."1995-96") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2000-01', "t0"."2000-01") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2005-06', "t0"."2005-06") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2010-11', "t0"."2010-11") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2011-12', "t0"."2011-12") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2012-13', "t0"."2012-13") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2013-14', "t0"."2013-14") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2014-15', "t0"."2014-15") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2015-16', "t0"."2015-16") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2016-17', "t0"."2016-17") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2017-18', "t0"."2017-18") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2018-19', "t0"."2018-19") AS Tuple(years Nullable(String), degrees Nullable(Int64))), CAST(tuple('2019-20', "t0"."2019-20") AS Tuple(years Nullable(String), degrees Nullable(Int64)))]
            ) AS "__pivoted__"
          FROM "humanities" AS "t0"
        ) AS "t1"
      ) AS "t2"
    ) AS "t3"
  ) AS "t4"
  GROUP BY
    "t4"."field_of_study"
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
    "t6"."diff" < 0
  ORDER BY
    "t6"."diff" ASC
  LIMIT 10
) AS "t11"