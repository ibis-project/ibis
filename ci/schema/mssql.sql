DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    carat REAL,
    cut NVARCHAR(MAX),
    color NVARCHAR(MAX),
    clarity NVARCHAR(MAX),
    depth REAL,
    "table" REAL,
    price BIGINT,
    x REAL,
    y REAL,
    z REAL
)

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    "playerID" NVARCHAR(255),
    "yearID" BIGINT,
    stint BIGINT,
    "teamID" NVARCHAR(7),
    "lgID" NVARCHAR(7),
    "G" BIGINT,
    "AB" BIGINT,
    "R" BIGINT,
    "H" BIGINT,
    "X2B" BIGINT,
    "X3B" BIGINT,
    "HR" BIGINT,
    "RBI" BIGINT,
    "SB" BIGINT,
    "CS" BIGINT,
    "BB" BIGINT,
    "SO" BIGINT,
    "IBB" BIGINT,
    "HBP" BIGINT,
    "SH" BIGINT,
    "SF" BIGINT,
    "GIDP" BIGINT
)

DROP TABLE IF EXISTS awards_players;

CREATE TABLE awards_players (
    "playerID" NVARCHAR(255),
    "awardID" NVARCHAR(255),
    "yearID" BIGINT,
    "lgID" NVARCHAR(7),
    tie NVARCHAR(7),
    notes NVARCHAR(255)
)

DROP TABLE IF EXISTS functional_alltypes;

CREATE TABLE functional_alltypes (
    "index" BIGINT,
    "Unnamed: 0" BIGINT,
    id INTEGER,
    bool_col BIT,
    tinyint_col SMALLINT, -- TINYINT is int64?
    smallint_col SMALLINT,
    int_col INTEGER,
    bigint_col BIGINT,
    float_col REAL,
    double_col DOUBLE PRECISION,
    date_string_col NVARCHAR(MAX),
    string_col NVARCHAR(MAX),
    timestamp_col DATETIME,
    year INTEGER,
    month INTEGER
)

CREATE INDEX "ix_functional_alltypes_index" ON functional_alltypes ("index");
