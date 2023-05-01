options (SKIP=1)
load data
  infile '/opt/oracle/csv/awards_players.csv'
  into table "awards_players"
  fields terminated by "," optionally enclosed by '"'
  TRAILING NULLCOLS
  ( "playerID", "awardID", "yearID", "lgID", "tie", "notes" )
