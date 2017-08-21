DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    `date` Date DEFAULT today(),
    carat Float64,
    cut String,
    color String,
    clarity String,
    depth Float64,
    `table` Float64,
    price Int64,
    x Float64,
    y Float64,
    z Float64
) ENGINE = MergeTree(date, (`carat`), 8192);

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    `date` Date DEFAULT today(),
    `playerID` String,
    `yearID` Int64,
    stint Int64,
    `teamID` String,
    `lgID` String,
    `G` Int64,
    `AB` Int64,
    `R` Int64,
    `H` Int64,
    `X2B` Int64,
    `X3B` Int64,
    `HR` Int64,
    `RBI` Int64,
    `SB` Int64,
    `CS` Int64,
    `BB` Int64,
    `SO` Int64,
    `IBB` Int64,
    `HBP` Int64,
    `SH` Int64,
    `SF` Int64,
    `GIDP` Int64
) ENGINE = MergeTree(date, (`playerID`), 8192);

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    `date` Date DEFAULT today(),
    `playerID` String,
    `awardID` String,
    `yearID` Int64,
    `lgID` String,
    tie String,
    notes String
) ENGINE = MergeTree(date, (`playerID`), 8192);

DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    `date` Date DEFAULT toDate(timestamp_col),
    `index` Int64,
    `Unnamed_0` Int64,
    id Int32,
    bool_col UInt8,
    tinyint_col Int16,
    smallint_col Int16,
    int_col Int32,
    bigint_col Int64,
    float_col Float32,
    double_col Float64,
    date_string_col String,
    string_col String,
    timestamp_col DateTime,
    year Int32,
    month Int32
) ENGINE = MergeTree(date, (`index`), 8192);

DROP TABLE IF EXISTS tzone;

CREATE TABLE tzone (
    `date` Date DEFAULT today(),
    ts DateTime,
    key String,
    value Float64
) ENGINE = MergeTree(date, (key), 8192);

DROP TABLE IF EXISTS array_types;

CREATE TABLE IF NOT EXISTS array_types (
    `date` Date DEFAULT today(),
    x Array(Int64),
    y Array(String),
    z Array(Float64),
    grouper String,
    scalar_column Float64
) ENGINE = MergeTree(date, (scalar_column), 8192);
