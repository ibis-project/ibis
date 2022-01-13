CREATE TABLE diamonds (
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
) ENGINE = Memory;

CREATE TABLE batting (
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
) ENGINE = Memory;

CREATE TABLE awards_players (
    `playerID` String,
    `awardID` String,
    `yearID` Int64,
    `lgID` String,
    tie String,
    notes String
) ENGINE = Memory;

CREATE TABLE functional_alltypes (
    `index` Int64,
    `Unnamed: 0` Int64,
    id Int32,
    bool_col UInt8,
    tinyint_col Int8,
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
) ENGINE = Memory;

CREATE TABLE tzone (
    ts DateTime,
    key String,
    value Float64
) ENGINE = Memory;

CREATE TABLE IF NOT EXISTS array_types (
    x Array(Int64),
    y Array(String),
    z Array(Float64),
    grouper String,
    scalar_column Float64
) ENGINE = Memory;
