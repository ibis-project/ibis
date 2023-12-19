SELECT
  parseDateTime64BestEffort('2015-01-01T12:34:56.789321+00:00', 6, 'UTC') AS "datetime.datetime(2015, 1, 1, 12, 34, 56, 789321, tzinfo=tzutc())"