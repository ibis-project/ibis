options (SKIP=1)
load data
  infile '/opt/oracle/data/diamonds.csv'
  into table "diamonds"
  fields csv without embedded
  ( "carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z" )
