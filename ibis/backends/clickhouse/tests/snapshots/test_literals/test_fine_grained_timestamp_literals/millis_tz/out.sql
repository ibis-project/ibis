SELECT
  parseDateTime64BestEffort('2015-01-01T12:34:56.789000+00:00', 3, 'UTC') AS "datetime.datetime(2015, 1, 1, 12, 34, 56, 789000, tzinfo=tzutc())"