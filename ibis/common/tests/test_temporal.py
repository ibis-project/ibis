from __future__ import annotations

import itertools
from datetime import date, datetime, time, timedelta, timezone

import dateutil
import pandas as pd
import pytest
import pytz
from packaging.version import parse as vparse
from pytest import param

from ibis.common.patterns import CoercedTo
from ibis.common.temporal import (
    DateUnit,
    IntervalUnit,
    TimeUnit,
    normalize_datetime,
    normalize_timedelta,
    normalize_timezone,
)
from ibis.conftest import WINDOWS

interval_units = pytest.mark.parametrize(
    ["singular", "plural", "short"],
    [
        ("year", "years", "Y"),
        ("quarter", "quarters", "Q"),
        ("month", "months", "M"),
        ("week", "weeks", "W"),
        ("day", "days", "D"),
        ("hour", "hours", "h"),
        ("minute", "minutes", "m"),
        ("second", "seconds", "s"),
        ("millisecond", "milliseconds", "ms"),
        ("microsecond", "microseconds", "us"),
        ("nanosecond", "nanoseconds", "ns"),
    ],
)


@interval_units
def test_interval_units(singular, plural, short):
    u = IntervalUnit[singular.upper()]
    assert u.singular == singular
    assert u.plural == plural
    assert u.short == short


@interval_units
def test_interval_unit_coercions(singular, plural, short):
    u = IntervalUnit[singular.upper()]
    v = CoercedTo(IntervalUnit)
    assert v.match(singular, {}) == u
    assert v.match(plural, {}) == u
    assert v.match(short, {}) == u


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("HH24", "h"),
        ("J", "D"),
        ("MI", "m"),
        ("SYYYY", "Y"),
        ("YY", "Y"),
        ("YYY", "Y"),
        ("YYYY", "Y"),
    ],
)
def test_interval_unit_aliases(alias, expected):
    v = CoercedTo(IntervalUnit)
    assert v.match(alias, {}) == IntervalUnit(expected)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1, IntervalUnit.DAY, 1),
        (1, IntervalUnit.HOUR, 1),
        (1, IntervalUnit.MINUTE, 1),
        (1, IntervalUnit.SECOND, 1),
        (1, IntervalUnit.MILLISECOND, 1),
        (1, IntervalUnit.MICROSECOND, 1),
        (timedelta(days=1), IntervalUnit.DAY, 1),
        (timedelta(hours=1), IntervalUnit.HOUR, 1),
        (timedelta(minutes=1), IntervalUnit.MINUTE, 1),
        (timedelta(seconds=1), IntervalUnit.SECOND, 1),
        (timedelta(milliseconds=1), IntervalUnit.MILLISECOND, 1),
        (timedelta(microseconds=1), IntervalUnit.MICROSECOND, 1),
        (timedelta(seconds=1, milliseconds=100), IntervalUnit.MILLISECOND, 1100),
        (timedelta(seconds=1, milliseconds=21), IntervalUnit.MICROSECOND, 1021000),
    ],
)
def test_normalize_timedelta(value, unit, expected):
    assert normalize_timedelta(value, unit) == expected


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        (timedelta(days=1), IntervalUnit.YEAR),
        (timedelta(days=1), IntervalUnit.QUARTER),
        (timedelta(days=1), IntervalUnit.MONTH),
        (timedelta(days=1), IntervalUnit.WEEK),
        (timedelta(hours=1), IntervalUnit.DAY),
        (timedelta(minutes=1), IntervalUnit.HOUR),
        (timedelta(seconds=1), IntervalUnit.MINUTE),
        (timedelta(milliseconds=1), IntervalUnit.SECOND),
        (timedelta(microseconds=1), IntervalUnit.MILLISECOND),
        (timedelta(days=1, microseconds=100), IntervalUnit.MILLISECOND),
    ],
)
def test_normalize_timedelta_invalid(value, unit):
    with pytest.raises(ValueError):
        normalize_timedelta(value, unit)


def test_interval_unit_compatibility():
    v = CoercedTo(IntervalUnit)
    for unit in itertools.chain(DateUnit, TimeUnit):
        interval = v.match(unit, {})
        assert isinstance(interval, IntervalUnit)
        assert unit.value == interval.value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (pytz.UTC, pytz.UTC),
        ("UTC", dateutil.tz.tzutc()),
        ("Europe/Budapest", dateutil.tz.gettz("Europe/Budapest")),
        (+2, timezone(timedelta(seconds=7200))),
        (-2, timezone(timedelta(seconds=-7200))),
        (dateutil.tz.tzoffset(None, 3600), timezone(timedelta(seconds=3600))),
    ],
)
def test_normalize_timezone(value, expected):
    assert normalize_timezone(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # datetime object
        (datetime(2017, 1, 1), datetime(2017, 1, 1)),
        (datetime(2017, 1, 1, 0, 0, 0, 1), datetime(2017, 1, 1, 0, 0, 0, 1)),
        (
            datetime(2017, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
            datetime(2017, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.UTC),
        ),
        # date object
        (datetime(2017, 1, 1).date(), datetime(2017, 1, 1)),
        # pandas timestamp object
        (pd.Timestamp("2017-01-01"), datetime(2017, 1, 1)),
        (pd.Timestamp("2017-01-01 00:00:00.000001"), datetime(2017, 1, 1, 0, 0, 0, 1)),
        # pandas timestamp object with timezone
        (
            pd.Timestamp("2017-01-01 00:00:00.000001+00:00"),
            datetime(2017, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.UTC),
        ),
        (
            pd.Timestamp("2017-01-01 00:00:00.000001+01:00"),
            datetime(2017, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.tzoffset(None, 3600)),
        ),
        # datetime string
        ("2017-01-01", datetime(2017, 1, 1)),
        ("2017-01-01 00:00:00.000001", datetime(2017, 1, 1, 0, 0, 0, 1)),
        # datetime string with timezone offset
        (
            "2017-01-01 00:00:00.000001+00:00",
            datetime(2017, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.UTC),
        ),
        (
            "2017-01-01 00:00:00.000001+01:00",
            datetime(2017, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.tzoffset(None, 3600)),
        ),
        # datetime string with timezone
        (
            "2017-01-01 00:00:00.000001 UTC",
            datetime(2017, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.UTC),
        ),
        (
            "2017-01-01 00:00:00.000001 GMT",
            datetime(2017, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.UTC),
        ),
        # plain integer
        (1000, datetime(1970, 1, 1, 0, 16, 40)),
        # floating point
        (1000.123, datetime(1970, 1, 1, 0, 16, 40, 123000)),
        # time object
        (time(0, 0, 0, 1), datetime.combine(date.today(), time(0, 0, 0, 1))),
    ],
)
def test_normalize_datetime(value, expected):
    result = normalize_datetime(value)
    assert result == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # timezone naive datetime
        (datetime(2017, 1, 1), None),
        (datetime(2017, 1, 1, 0, 0, 0, 1), None),
        # timezone aware datetime
        (datetime(2022, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.UTC), "UTC"),
        # timezone aware datetime with timezone offset
        (
            datetime(2022, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.tzoffset(None, 3600)),
            "UTC+01:00",
        ),
        # timezone aware datetime with timezone name
        (datetime(2022, 1, 1, 0, 0, 0, 1, tzinfo=dateutil.tz.gettz("CET")), "CET"),
        # pandas timestamp with timezone
        (pd.Timestamp("2022-01-01 00:00:00.000001+00:00"), "UTC"),
        param(
            pd.Timestamp("2022-01-01 00:00:00.000001+01:00"),
            "UTC+01:00",
            marks=pytest.mark.xfail(
                vparse(pd.__version__) < vparse("2.0.0") and not WINDOWS,
                reason=(
                    "tzdata is missing in pandas < 2.0.0 due to an incorrect marker "
                    "in the tzdata package specification that restricts its installation "
                    "to windows only"
                ),
            ),
        ),
    ],
)
def test_normalized_datetime_tzname(value, expected):
    result = normalize_datetime(value)
    assert result.tzname() == expected
