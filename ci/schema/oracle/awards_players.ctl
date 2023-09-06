options (SKIP=1)
load data
  infile '/opt/oracle/data/awards_players.csv'
  into table "awards_players"
  fields csv without embedded
  TRAILING NULLCOLS
  ( "playerID", "awardID", "yearID", "lgID", "tie", "notes" )
