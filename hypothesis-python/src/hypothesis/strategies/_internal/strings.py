# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import copy
import re
import warnings
from functools import lru_cache, partial
from typing import Optional

from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal import charmap
from hypothesis.internal.conjecture.data import COLLECTION_DEFAULT_MAX_SIZE
from hypothesis.internal.filtering import max_len, min_len
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.strategies._internal.collections import ListStrategy
from hypothesis.strategies._internal.lazy import unwrap_strategies
from hypothesis.strategies._internal.numbers import IntegersStrategy
from hypothesis.strategies._internal.strategies import (
    OneOfStrategy,
    SampledFromStrategy,
    SearchStrategy,
)
from hypothesis.vendor.pretty import pretty
import hypothesis


# Cache size is limited by sys.maxunicode, but passing None makes it slightly faster.
@lru_cache(maxsize=None)
def _check_is_single_character(c):
    # In order to mitigate the performance cost of this check, we use a shared cache,
    # even at the cost of showing the culprit strategy in the error message.
    if not isinstance(c, str):
        type_ = get_pretty_function_description(type(c))
        raise InvalidArgument(f"Got non-string {c!r} (type {type_})")
    if len(c) != 1:
        raise InvalidArgument(f"Got {c!r} (length {len(c)} != 1)")
    return c


class OneCharStringStrategy(SearchStrategy):
    """A strategy which generates single character strings of text type."""

    def __init__(self, intervals, force_repr=None):
        assert isinstance(intervals, IntervalSet)
        self.intervals = intervals
        self._force_repr = force_repr

    @classmethod
    def from_characters_args(
        cls,
        *,
        codec=None,
        min_codepoint=None,
        max_codepoint=None,
        categories=None,
        exclude_characters=None,
        include_characters=None,
    ):
        assert set(categories or ()).issubset(charmap.categories())
        intervals = charmap.query(
            min_codepoint=min_codepoint,
            max_codepoint=max_codepoint,
            categories=categories,
            exclude_characters=exclude_characters,
            include_characters=include_characters,
        )
        if codec is not None:
            intervals &= charmap.intervals_from_codec(codec)
        _arg_repr = ", ".join(
            f"{k}={v!r}"
            for k, v in [
                ("codec", codec),
                ("min_codepoint", min_codepoint),
                ("max_codepoint", max_codepoint),
                ("categories", categories),
                ("exclude_characters", exclude_characters),
                ("include_characters", include_characters),
            ]
            if v not in (None, "", set(charmap.categories()) - {"Cs"})
        )
        if not intervals:
            raise InvalidArgument(
                "No characters are allowed to be generated by this "
                f"combination of arguments: {_arg_repr}"
            )
        return cls(intervals, force_repr=f"characters({_arg_repr})")

    @classmethod
    def from_alphabet(cls, alphabet):
        if isinstance(alphabet, str):
            return cls.from_characters_args(categories=(), include_characters=alphabet)

        assert isinstance(alphabet, SearchStrategy)
        char_strategy = unwrap_strategies(alphabet)
        if isinstance(char_strategy, cls):
            return char_strategy
        elif isinstance(char_strategy, SampledFromStrategy):
            for c in char_strategy.elements:
                _check_is_single_character(c)
            return cls.from_characters_args(
                categories=(),
                include_characters=char_strategy.elements,
            )
        elif isinstance(char_strategy, OneOfStrategy):
            intervals = IntervalSet()
            for s in char_strategy.element_strategies:
                intervals = intervals.union(cls.from_alphabet(s).intervals)
            return cls(intervals, force_repr=repr(alphabet))
        raise InvalidArgument(
            f"{alphabet=} must be a sampled_from() or characters() strategy"
        )

    def __repr__(self):
        return self._force_repr or f"OneCharStringStrategy({self.intervals!r})"

    def do_draw(self, data):
        return data.draw_string(self.intervals, min_size=1, max_size=1)


_nonempty_names = (
    "capitalize",
    "expandtabs",
    "join",
    "lower",
    "rsplit",
    "split",
    "splitlines",
    "swapcase",
    "title",
    "upper",
)
_nonempty_and_content_names = (
    "islower",
    "isupper",
    "isalnum",
    "isalpha",
    "isascii",
    "isdigit",
    "isspace",
    "istitle",
    "lstrip",
    "rstrip",
    "strip",
)


class TextStrategy(ListStrategy):
    def do_draw(self, data):
        if not hypothesis.fuzzing.global_fuzzing_use_ir:
            return "".join(super().do_draw(data))

        # if our element strategy is OneCharStringStrategy, we can skip the
        # ListStrategy draw and jump right to our nice IR string draw.
        # Doing so for user-provided element strategies is not correct in
        # general, as they may define a different distribution than our IR.
        elems = unwrap_strategies(self.element_strategy)
        if isinstance(elems, OneCharStringStrategy):
            return data.draw_string(
                elems.intervals,
                min_size=self.min_size,
                max_size=(
                    COLLECTION_DEFAULT_MAX_SIZE
                    if self.max_size == float("inf")
                    else self.max_size
                ),
            )
        return "".join(super().do_draw(data))

    def __repr__(self):
        args = []
        if repr(self.element_strategy) != "characters()":
            args.append(repr(self.element_strategy))
        if self.min_size:
            args.append(f"min_size={self.min_size}")
        if self.max_size < float("inf"):
            args.append(f"max_size={self.max_size}")
        return f"text({', '.join(args)})"

    # See https://docs.python.org/3/library/stdtypes.html#string-methods
    # These methods always return Truthy values for any nonempty string.
    _nonempty_filters = (
        *ListStrategy._nonempty_filters,
        str,
        str.casefold,
        str.encode,
        *(getattr(str, n) for n in _nonempty_names),
    )
    _nonempty_and_content_filters = (
        str.isdecimal,
        str.isnumeric,
        *(getattr(str, n) for n in _nonempty_and_content_names),
    )

    def filter(self, condition):
        elems = unwrap_strategies(self.element_strategy)
        if (
            condition is str.isidentifier
            and self.max_size >= 1
            and isinstance(elems, OneCharStringStrategy)
        ):
            from hypothesis.strategies import builds, nothing

            id_start, id_continue = _identifier_characters()
            if not (elems.intervals & id_start):
                return nothing()
            return builds(
                "{}{}".format,
                OneCharStringStrategy(elems.intervals & id_start),
                TextStrategy(
                    OneCharStringStrategy(elems.intervals & id_continue),
                    min_size=max(0, self.min_size - 1),
                    max_size=self.max_size - 1,
                ),
                # Filter to ensure that NFKC normalization keeps working in future
            ).filter(str.isidentifier)
        if (new := _string_filter_rewrite(self, str, condition)) is not None:
            return new
        return super().filter(condition)


def _string_filter_rewrite(self, kind, condition):
    if condition in (kind.lower, kind.title, kind.upper):
        k = kind.__name__
        warnings.warn(
            f"You applied {k}.{condition.__name__} as a filter, but this allows "
            f"all nonempty strings!  Did you mean {k}.is{condition.__name__}?",
            HypothesisWarning,
            stacklevel=2,
        )

    if (
        (
            kind is bytes
            or isinstance(
                unwrap_strategies(self.element_strategy), OneCharStringStrategy
            )
        )
        and isinstance(pattern := getattr(condition, "__self__", None), re.Pattern)
        and isinstance(pattern.pattern, kind)
    ):
        from hypothesis.strategies._internal.regex import regex_strategy

        if condition.__name__ == "match":
            # Replace with an easier-to-handle equivalent condition
            caret, close = ("^(?:", ")") if kind is str else (b"^(?:", b")")
            pattern = re.compile(caret + pattern.pattern + close, flags=pattern.flags)
            condition = pattern.search

        if condition.__name__ in ("search", "findall", "fullmatch"):
            s = regex_strategy(
                pattern,
                fullmatch=condition.__name__ == "fullmatch",
                alphabet=self.element_strategy if kind is str else None,
            )
            if self.min_size > 0:
                s = s.filter(partial(min_len, self.min_size))
            if self.max_size < 1e999:
                s = s.filter(partial(max_len, self.max_size))
            return s
        elif condition.__name__ in ("finditer", "scanner"):
            # PyPy implements `finditer` as an alias to their `scanner` method
            warnings.warn(
                f"You applied {pretty(condition)} as a filter, but this allows "
                f"any string at all!  Did you mean .findall ?",
                HypothesisWarning,
                stacklevel=3,
            )
            return self
        elif condition.__name__ == "split":
            warnings.warn(
                f"You applied {pretty(condition)} as a filter, but this allows "
                f"any nonempty string!  Did you mean .search ?",
                HypothesisWarning,
                stacklevel=3,
            )
            return self.filter(bool)

    # We use ListStrategy filter logic for the conditions that *only* imply
    # the string is nonempty.  Here, we increment the min_size but still apply
    # the filter for conditions that imply nonempty *and specific contents*.
    if condition in self._nonempty_and_content_filters and self.max_size >= 1:
        self = copy.copy(self)
        self.min_size = max(1, self.min_size)
        return ListStrategy.filter(self, condition)

    return None


# Excerpted from https://www.unicode.org/Public/15.0.0/ucd/PropList.txt
# Python updates it's Unicode version between minor releases, but fortunately
# these properties do not change between the Unicode versions in question.
_PROPLIST = """
# ================================================

1885..1886    ; Other_ID_Start # Mn   [2] MONGOLIAN LETTER ALI GALI BALUDA..MONGOLIAN LETTER ALI GALI THREE BALUDA
2118          ; Other_ID_Start # Sm       SCRIPT CAPITAL P
212E          ; Other_ID_Start # So       ESTIMATED SYMBOL
309B..309C    ; Other_ID_Start # Sk   [2] KATAKANA-HIRAGANA VOICED SOUND MARK..KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK

# Total code points: 6

# ================================================

00B7          ; Other_ID_Continue # Po       MIDDLE DOT
0387          ; Other_ID_Continue # Po       GREEK ANO TELEIA
1369..1371    ; Other_ID_Continue # No   [9] ETHIOPIC DIGIT ONE..ETHIOPIC DIGIT NINE
19DA          ; Other_ID_Continue # No       NEW TAI LUE THAM DIGIT ONE

# Total code points: 12
"""


@lru_cache
def _identifier_characters():
    """See https://docs.python.org/3/reference/lexical_analysis.html#identifiers"""
    # Start by computing the set of special characters
    chars = {"Other_ID_Start": "", "Other_ID_Continue": ""}
    for line in _PROPLIST.splitlines():
        if m := re.match(r"([0-9A-F.]+) +; (\w+) # ", line):
            codes, prop = m.groups()
            span = range(int(codes[:4], base=16), int(codes[-4:], base=16) + 1)
            chars[prop] += "".join(chr(x) for x in span)

    # Then get the basic set by Unicode category and known extras
    id_start = charmap.query(
        categories=("Lu", "Ll", "Lt", "Lm", "Lo", "Nl"),
        include_characters="_" + chars["Other_ID_Start"],
    )
    id_start -= IntervalSet.from_string(
        # Magic value: the characters which NFKC-normalize to be invalid identifiers.
        # Conveniently they're all in `id_start`, so we only need to do this once.
        "\u037a\u0e33\u0eb3\u2e2f\u309b\u309c\ufc5e\ufc5f\ufc60\ufc61\ufc62\ufc63"
        "\ufdfa\ufdfb\ufe70\ufe72\ufe74\ufe76\ufe78\ufe7a\ufe7c\ufe7e\uff9e\uff9f"
    )
    id_continue = id_start | charmap.query(
        categories=("Mn", "Mc", "Nd", "Pc"),
        include_characters=chars["Other_ID_Continue"],
    )
    return id_start, id_continue


class BytesStrategy(SearchStrategy):
    def __init__(self, min_size: int, max_size: Optional[int]):
        self.min_size = min_size
        self.max_size = (
            max_size if max_size is not None else COLLECTION_DEFAULT_MAX_SIZE
        )

    def do_draw(self, data):
        if not hypothesis.fuzzing.global_fuzzing_use_ir:
            s = ListStrategy(
                IntegersStrategy(0, 255), min_size=self.min_size, max_size=self.max_size
            )
            if self.min_size == self.max_size:
                return bytes(data.draw_bytes(self.min_size, self.max_size))
            return bytes(s.do_draw(data))

        return data.draw_bytes(self.min_size, self.max_size)

    _nonempty_filters = (
        *ListStrategy._nonempty_filters,
        bytes,
        *(getattr(bytes, n) for n in _nonempty_names),
    )
    _nonempty_and_content_filters = (
        *(getattr(bytes, n) for n in _nonempty_and_content_names),
    )

    def filter(self, condition):
        if (new := _string_filter_rewrite(self, bytes, condition)) is not None:
            return new
        return ListStrategy.filter(self, condition)
