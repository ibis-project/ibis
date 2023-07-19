from __future__ import annotations

import ast
import re

import parsy

_STRING_REGEX = (
    """('[^\n'\\\\]*(?:\\\\.[^\n'\\\\]*)*'|"[^\n"\\\\"]*(?:\\\\.[^\n"\\\\]*)*")"""
)

SPACES = parsy.regex(r'\s*', re.MULTILINE)


def spaceless(parser):
    return SPACES.then(parser).skip(SPACES)


def spaceless_string(*strings: str):
    return spaceless(
        parsy.alt(*(parsy.string(string, transform=str.lower) for string in strings))
    )


SINGLE_DIGIT = parsy.decimal_digit
RAW_NUMBER = SINGLE_DIGIT.at_least(1).concat()
PRECISION = SCALE = NUMBER = LENGTH = RAW_NUMBER.map(int)
TEMPORAL_SCALE = SINGLE_DIGIT.map(int)

LPAREN = spaceless_string("(")
RPAREN = spaceless_string(")")

LBRACKET = spaceless_string("[")
RBRACKET = spaceless_string("]")

LANGLE = spaceless_string("<")
RANGLE = spaceless_string(">")

COMMA = spaceless_string(",")
COLON = spaceless_string(":")
SEMICOLON = spaceless_string(";")

RAW_STRING = parsy.regex(_STRING_REGEX).map(ast.literal_eval)
FIELD = parsy.regex("[a-zA-Z_][a-zA-Z_0-9]*")
