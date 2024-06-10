from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pandas._libs.parsers import STR_NA_VALUES

import ibis
import ibis.expr.types as ir
from ibis import _

if TYPE_CHECKING:
    from collections.abc import Iterable

# STR_NA_VALUES is a set of strings to interpret as nan, such as
# {'null', '-1.#QNAN', '#N/A', '', '#N/A N/A', 'n/a', 'nan', 'NaN', ...}
# Add our own strings here.
NAN_LIKE = STR_NA_VALUES | {
    "anonymous",
    "anon",
    "need",
    "none",
    "not known",
    "not provided",
    "not reported",
    "not specified",
    "pending",
    "unknown",
    "unidentified",
}
NAN_LIKE = {s.lower() for s in NAN_LIKE}


def contains_alphanum(col: ir.StringValue) -> ir.BooleanValue:
    return col.lower().re_search(r".*[a-z0-9].*")


def is_nan_like(col: ir.Value) -> ir.BooleanValue:
    if not col.type().is_string():
        return col.isnull()
    result = col.isnull()
    result |= col.lower().isin(NAN_LIKE).fill_null(False)
    result |= ~contains_alphanum(col).fill_null(False)
    return result


def normalize_nulls(data: ir.StringValue) -> ir.StringValue:
    """Fill any nan-likes with NULL."""
    return is_nan_like(data).ifelse(ibis.null(), data)


def starts_with(a: ir.StringValue, b: ir.StringValue) -> ir.BooleanValue:
    """Apply a.startswith(b) to each pair of values in a and b."""
    return a.like(b + "%")


def strip(s: ir.StringValue, chars: Iterable[str]) -> ir.StringValue:
    """Remove leading and trailing characters from a string."""

    def _strip_pattern(chars: Iterable[str]) -> str:
        # If chars contains a "(" or some other regex keyword, we want
        # it to match a literal "("
        chars = (re.escape(c) for c in chars)
        char_pattern = "|".join(chars)
        char_pattern = f"({char_pattern})+"
        return char_pattern

    s = s.cast("string")
    return s.re_replace(
        "^" + _strip_pattern(chars) + "|" + _strip_pattern(chars) + "$",
        "",
    )


def fix_nicknames(t: ir.Table) -> ir.Table:
    """Splits "Nicholas (Nick)" into "Nicholas" and "Nick".

    Also deals with the form "Nicholas 'Nick'" etc.
    """
    first, nick = _parse_nicknames(t["first_name"], t["nickname"])
    return t.mutate(first_name=first, nickname=nick)


def _parse_nicknames(
    first: ir.StringValue, nick: ir.StringValue
) -> tuple[ir.StringValue, ir.StringValue]:
    enclosures = [
        r"'.*'",
        r'".*"',
        r"\(.*\)",
    ]
    enclosure_pattern = "|".join(enclosures)
    pattern = "(.*)(" + enclosure_pattern + ")"
    # 1 based
    before = first.re_extract(pattern, 1)
    inside = first.re_extract(pattern, 2)
    inside = strip(inside, "'\"()")
    before = norm_whitespace(before)
    inside = norm_whitespace(inside)

    # names_before = spacy.names_in_texts(before)
    # names_inside = spacy.names_in_texts(inside)
    # nb = names_before.apply(lambda x: x[0] if len(x) > 0 else pd.NA)
    # ni = names_inside.apply(lambda x: x[0] if len(x) > 0 else pd.NA)
    # fixable = nb.isna() & is_single_name_series(inside)
    # nb[fixable] = inside[fixable]

    needs_fix = first.notnull() & nick.isnull()
    before_good = before.split(" ").length() == 1
    inside_good = inside.split(" ").length() == 1

    todo = needs_fix & before_good & inside_good
    return todo.ifelse(before, first), todo.ifelse(inside, nick)


def to_alphanum(s: ir.StringValue, fill: str = " ") -> ir.StringValue:
    s = to_ascii(s)
    s = s.re_replace(r"[^A-Za-z0-9]+", fill)
    s = norm_whitespace(s)
    return s


def norm_whitespace(s: ir.StringValue) -> ir.StringValue:
    """Convert all whitespace to a single space, strip leading/trailing spaces."""
    s = s.cast("string")
    s = s.re_replace(r"\s+", " ")
    s = s.strip()
    s = s.nullif("")
    return s


def to_ascii(s: ir.StringValue) -> ir.StringValue:
    """Remove any non-ascii characters."""
    # return norm_whitespace(s.fill_null("").apply(unidecode).astype(s.dtype))
    # We don't have access to the unidecode function, so just strip out
    # non-ascii characters
    s = s.cast("string")
    return s.re_replace(r"[^\x00-\x7F]+", "")


def num_tokens(s: ir.StringValue) -> ir.IntegerValue:
    """Count the number of tokens, separated by spaces, in a string."""
    s = s.cast("string")
    s = s.re_replace(r"\s+", " ")
    s = s.strip()
    s = s.nullif("")
    return s.split(" ").length().fill_null(0)


NAME_COLUMNS = [
    "prefix",
    "first_name",
    "middle_name",
    "last_name",
    "suffix",
    "nickname",
]


# It would be possible to do something more sophisticated here to determine
# which of first or last to clear. But I wanted to keep this so it only operates
# on a per-row basis. This prevents us from looking atir.Table-wide statistics
# on the names to determine if a name is more likely a first or last name,
# because then the result of one reow would depend on the rest of the rows.
# In the future we could use an external lookupir.Table of likelihoods.
def drop_first_when_same_as_last(t: ir.Table) -> ir.Table:
    """Where the f and l are each one token and the same, set the first to null."""

    def _norm(s: ir.StringColumn) -> ir.StringColumn:
        s = to_alphanum(s)
        s = s.lower()
        s = s.strip()
        return s

    fn = _norm(t.first_name)
    ln = _norm(t.last_name)
    todo = ibis.and_(
        fn == ln,
        num_tokens(fn) == 1,
        num_tokens(ln) == 1,
    )
    return t.mutate(first_name=todo.ifelse(ibis.null(), t.first_name))


def fix_duplicate_appearances(t: ir.Table) -> ir.Table:
    """If a field appears as a token in another field, remove it from the other field.

    for example:
    >>> names = pd.DataFrame(
    ...     [
    ...         {"first_name": "J ang", "middle_name": "A Ang", "last_name": "ANG"},
    ...         {"first_name": "smith john", "middle_name": "B", "last_name": "SMITH"},
    ...     ]
    ... )
    >>> scrub_duplicate_appearances(names, "last_name")
    pd.DataFrame(
        [
            {'first_name': 'J', 'middle_name': 'A', 'last_name': 'ANG'},
            {'first_name': 'john', 'middle_name': 'B', 'last_name': 'SMITH'},
        ]
    )
    """
    SEP = " "
    token_cols = {c + "_tokens": _[c].split(SEP) for c in NAME_COLUMNS}
    t2 = t.mutate(**token_cols)

    def single_or_null(col):
        # If a column is a single token, choose it. Otherwise null.
        is_single = _[col + "_tokens"].length() == 1
        return is_single.ifelse(_[col].upper(), ibis.null())

    single_or_null_cols = {c + "_single": single_or_null(c) for c in NAME_COLUMNS}
    t3 = t2.mutate(**single_or_null_cols)

    def singles_except(col):
        # merge all columns into a single column of type array
        # e.g.
        # prefix_single first_single middle_single last_single
        #  a               b           null            d
        # singles_except('prefix') -> ['b', null, 'd']
        all_but_c = (c + "_single" for c in NAME_COLUMNS if c != col)
        return ibis.array([t3[c] for c in all_but_c])

    se = {"singles_except_" + c: singles_except(c) for c in NAME_COLUMNS}
    t4 = t3.mutate(**se)

    def filter_tokens(t, col):
        # can't use Deferreds until this bug fixed:
        # https://github.com/ibis-project/ibis/issues/7626
        return t[col + "_tokens"].filter(
            lambda token: ~t[f"singles_except_{col}"].contains(token.upper())
        )

    tokens_filtered = {
        c + "_tokens_filtered": filter_tokens(t4, c) for c in NAME_COLUMNS
    }
    t5 = t4.mutate(**tokens_filtered)

    final_changes = {c: _[c + "_tokens_filtered"].join(SEP) for c in NAME_COLUMNS}
    return t5.mutate(**final_changes).select(*t.columns)


def choose_longer(s1: ir.StringColumn, s2: ir.StringColumn) -> ir.StringColumn:
    l1 = s1.length().fill_null(0)
    l2 = s2.length().fill_null(0)
    return (l1 > l2).ifelse(s1, s2)


def parse_middle(
    first: ir.StringColumn, middle: ir.StringColumn, last: ir.StringColumn
) -> tuple[ir.StringColumn, ir.StringColumn, ir.StringColumn]:
    """Extract the middle name from the first or last name field.

    Four kinds of parsings
    A jones     -> Parse
    AJ          -> Inconclusive, could be first name.
    A J         -> Inconclusive, could be first name.
    Alice J     -> Parse
    Alice Jones -> Inconclusive, could be first name.

    Only can be confident in this if there are two tokens, and at least one of
    them is only a single letter (optionally followed by period).

    Two capital letters in the first name is not sufficient, for example TJ
    is probably the first name.

    Also looks at if middle_name is already filled.

    Misses some cases that we could worry about, but this gets the vast majority.
    """
    idx = first.notnull() & middle.isnull()

    # if pd.notna(name.middle_name) or pd.isna(name.first_name):
    #     return name
    pattern = r"^\s*(\w+)\.?\s+(\w+)\.?\s*$"
    a = first.re_extract(pattern, 1).nullif("")
    b = first.re_extract(pattern, 2).nullif("")

    # Deal with "Kay Ellen", "E" should yield "Kay", "Ellen"
    middle_is_middle = (starts_with(middle, b) | starts_with(b, middle)).fill_null(
        False
    )
    result_first = middle_is_middle.ifelse(a, first)
    result_middle = middle_is_middle.ifelse(choose_longer(b, middle), middle)

    al = a.length().fill_null(0)
    bl = b.length().fill_null(0)
    short_long = (al == 1) & (bl > 1)  # A Jones
    long_short = (al > 1) & (bl == 1)  # Alice J
    idx &= short_long | long_short

    # Many rows are of the form first_name="H Daniel", last_name="Hull"
    # where the first token of the first name is actually the
    # first letter of the last name. Catch this.
    first_is_last = starts_with(last, a).fill_null(False)
    fil = idx & first_is_last
    result_first = fil.ifelse(b, result_first)
    result_middle = fil.ifelse(ibis.null(), result_middle)

    finl = idx & ~first_is_last
    result_first = finl.ifelse(a, result_first)
    result_middle = finl.ifelse(b, result_middle)

    # Correct for when the last name is "A Jones"
    a = last.re_extract(pattern, 1).nullif("")
    b = last.re_extract(pattern, 2).nullif("")
    al = a.length().fill_null(0)
    bl = b.length().fill_null(0)
    idx = (al == 1) & (bl > 1)  # A Jones
    idx &= middle.isnull()
    result_middle = idx.ifelse(a, result_middle)
    result_last = idx.ifelse(b, last)

    return result_first, result_middle, result_last


def fix_middle(names: ir.Table) -> ir.Table:
    f, m, la = parse_middle(
        names["first_name"], names["middle_name"], names["last_name"]
    )
    return names.mutate(first_name=f, middle_name=m, last_name=la)


def fix_nickname_is_middle(t: ir.Table) -> ir.Table:
    """Fix 'george "mike" m smith", Where mike is both the nickname and middle name.

    Watch out for when the nickname is probably not related to the middle name,
    Such as with 'Carolyn "Care" c smith' (Care is short for Carolyn, not the middle)
    """
    todo = starts_with(t["nickname"], t["middle_name"]).fill_null(False)
    # Get rid of the 'Carolyn "Care" c smith' case
    todo &= ~starts_with(t["first_name"], t["middle_name"])
    return t.mutate(middle_name=todo.ifelse(t.nickname, t.middle_name))


def fix_two_token_people(t: ir.Table) -> ir.Table:
    """If no first name, and a two-token last name, extract first and last into respective columns."""
    first, last = _extract_two_token_people(t.first_name, t.last_name)
    return t.mutate(first_name=first, last_name=last)


def _extract_two_token_people(
    first: ir.StringColumn, last: ir.StringColumn
) -> tuple[ir.StringColumn, ir.StringColumn]:
    first = first.nullif("")
    last = last.nullif("")
    tokens = last.split(" ")
    fi = norm_whitespace(tokens[0])
    la = norm_whitespace(tokens[1])
    idx = last.notnull() & first.isnull() & (tokens.length() == 2)
    return idx.ifelse(fi, first), idx.ifelse(la, last)


def fix_last_comma_first(t: ir.Table) -> ir.Table:
    """Parses when first name is NA, and last name follows the form "Last, First"."""
    pattern = r"^(.*),(.*)$"
    a = t.last_name.re_extract(pattern, 1)
    b = t.last_name.re_extract(pattern, 2)
    a = norm_whitespace(a)
    b = norm_whitespace(b)
    one_each = (num_tokens(a) == 1) & (num_tokens(b) == 1)
    first_empty = t.first_name.strip().fill_null("") == ""
    todo = first_empty & one_each
    return t.mutate(
        first_name=todo.ifelse(b, t.first_name),
        last_name=todo.ifelse(a, t.last_name),
    )


def norm_name_fields(t: ir.Table) -> ir.Table:
    """Normalize capitalization and whitespace, remove weird punctuation, etc."""
    t = t.mutate(**{col: norm_name_field(t[col]) for col in NAME_COLUMNS})
    # suffixes have some specialcapitalization rules
    t = t.mutate(prefix=_norm_affix(t.prefix), suffix=_norm_affix(t.suffix))
    return t


def norm_name_field(s: ir.StringValue) -> ir.StringValue:
    """Fix capitalization and whitespace in a name field.

    "DEBOER" -> "Deboer"
    "DeBoer" -> "DeBoer"
    "OBrien" -> "OBrien"
    "O'Brien" -> "O'Brien"
    "Jr. " -> "Jr"
    "Abc -Def" -> "Abc-Def"
    """

    def _norm_token(t: ir.StringValue) -> ir.StringValue:
        num_upper = t.re_replace(r"[^A-Z]", "").length()
        num_lower = t.re_replace(r"[^a-z]", "").length()
        multi_caps = (num_upper >= 2) & (num_lower >= 1)
        caps_in_middler = (num_upper >= 1) & t.re_search(r"[a-z]")
        purposefully_weird = multi_caps | caps_in_middler
        return purposefully_weird.ifelse(t, t.capitalize())

    s = s.re_replace(r"[^a-zA-Z \-']", "")
    s = s.split(" ").map(_norm_token).join(" ")
    s = s.re_replace(r"\s+", " ")
    s = s.re_replace(r"(\W) (\W)", r"\1\2")  # between non-words
    s = s.re_replace(r"(\w) (\W)", r"\1\2")  # between word and non-word
    s = s.re_replace(r"(\W) (\w)", r"\1\2")  # between non-word and word
    s = s.strip()
    return s


def _norm_affix(s: ir.StringValue) -> ir.StringValue:
    s = s.upper()
    s = s.substitute(
        {
            "JR": "Jr",
            "SR": "Sr",
            "PHD": "PhD",
            "MR": "Mr",
            "MRS": "Mrs",
            "MS": "Ms",
            "DR": "Dr",
        }
    )
    return s


def clean_names(names: ir.Table) -> ir.Table:
    """Perform all the cleaning steps on a Table of names."""
    names = names.mutate(**{col: normalize_nulls(names[col]) for col in NAME_COLUMNS})
    # names = names.cache()
    names = fix_nicknames(names)
    # names = names.cache()
    names = fix_last_comma_first(names)
    # names = names.cache()
    names = fix_two_token_people(names)
    # names = names.cache()
    names = fix_middle(names)
    # names = names.cache()
    names = drop_first_when_same_as_last(names)
    # names = names.cache()
    names = fix_duplicate_appearances(names)
    # names = names.cache()
    names = norm_name_fields(names)
    # names = names.cache()
    names = fix_nickname_is_middle(names)
    # names = names.cache()
    return names
