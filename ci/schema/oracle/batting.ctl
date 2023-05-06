options (SKIP=1)
load data
  infile '/opt/oracle/data/batting.csv'
  into table "batting"
  fields terminated by "," optionally enclosed by '"'
  TRAILING NULLCOLS
  ( "playerID", "yearID", "stint", "teamID", "lgID", "G", "AB", "R", "H", "X2B", "X3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP" )
