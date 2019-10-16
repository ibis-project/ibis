DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds (
    carat REAL,
    cut NTEXT,
    color NTEXT,
    clarity NTEXT,
    depth REAL,
    "table" REAL,
    price BIGINT,
    x REAL,
    y REAL,
    z REAL
)

DROP TABLE IF EXISTS batting;

CREATE TABLE batting (
    "playerID" VARCHAR(255),
    "yearID" BIGINT,
    stint BIGINT,
    "teamID" VARCHAR(7),
    "lgID" VARCHAR(7),
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
    "playerID" VARCHAR(255),
    "awardID" VARCHAR(255),
    "yearID" BIGINT,
    "lgID" VARCHAR(7),
    tie VARCHAR(7),
    notes VARCHAR(255)
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
    date_string_col NTEXT,
    string_col NTEXT,
    timestamp_col DATETIME,
    year INTEGER,
    month INTEGER
)

CREATE INDEX "ix_functional_alltypes_index" ON functional_alltypes ("index");
