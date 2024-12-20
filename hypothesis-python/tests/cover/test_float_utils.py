# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
from sys import float_info

import pytest

from hypothesis import assume, example, given, strategies as st
from hypothesis.internal.floats import (
    SMALLEST_SUBNORMAL,
    count_between_floats,
    make_float_clamper,
    next_down,
    next_up,
    sign_aware_lte,
)


def test_can_handle_straddling_zero():
    assert count_between_floats(-0.0, 0.0) == 2


@pytest.mark.parametrize(
    "func,val",
    [
        (next_up, math.nan),
        (next_up, math.inf),
        (next_up, -0.0),
        (next_down, math.nan),
        (next_down, -math.inf),
        (next_down, 0.0),
    ],
)
def test_next_float_equal(func, val):
    if math.isnan(val):
        assert math.isnan(func(val))
    else:
        assert func(val) == val


# exponent comparisons:
@example(1, float_info.max, 0, True)
@example(1, float_info.max, 1, True)
@example(1, float_info.max, 10, True)
@example(1, float_info.max, float_info.max, True)
@example(1, float_info.max, math.inf, True)
# mantissa comparisons:
@example(100.0001, 100.0003, 100.0001, True)
@example(100.0001, 100.0003, 100.0002, True)
@example(100.0001, 100.0003, 100.0003, True)
@given(st.floats(), st.floats(), st.floats(), st.booleans())
def test_float_clamper(min_value, max_value, input_value, allow_nan):
    assume(sign_aware_lte(min_value, max_value))

    clamper = make_float_clamper(min_value, max_value, SMALLEST_SUBNORMAL, allow_nan)
    clamped = clamper(input_value)
    if math.isnan(clamped):
        # if we clamped to nan, we should be allowing nan.
        assert allow_nan
    else:
        # otherwise, we should have clamped to something in the permitted range.
        assert sign_aware_lte(min_value, clamped) and sign_aware_lte(clamped, max_value)

    # if input_value was permitted in the first place, then the clamped value should
    # be the same as the input value.
    if sign_aware_lte(min_value, input_value) and sign_aware_lte(
        input_value, max_value
    ):
        assert input_value == clamped
