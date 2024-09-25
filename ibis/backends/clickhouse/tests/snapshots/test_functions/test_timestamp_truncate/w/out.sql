SELECT
  CAST(toMonday(parseDateTimeBestEffort('2009-05-17T12:34:56')) AS Nullable(DateTime)) AS "TimestampTruncate(datetime.datetime(2009, 5, 17, 12, 34, 56), WEEK)"