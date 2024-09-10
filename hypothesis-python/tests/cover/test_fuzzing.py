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
import random
import string
import uuid
from random import Random

import pytest

from hypothesis import assume, example, given, settings, strategies as st
from hypothesis.core import BUFFER_SIZE, custom_mutator, mutate_string
from hypothesis.internal.floats import SMALLEST_SUBNORMAL, next_down, next_up
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.internal.reflection import get_pretty_function_description

from tests.common.utils import flaky
from tests.conjecture.common import run_to_buffer
from tests.common.strategies import intervals

MARKER = uuid.uuid4().hex

# stop hypothesis from seeding our random
r = Random(random.randint(0, int(1e10)))


def fuzz(f, *, start, mode, max_examples):
    fuzz_one_input = f.hypothesis._get_fuzz_target(
        args=(), kwargs={}, use_atheris=mode == "atheris"
    )
    fuzz_one_input(start)
    for _ in range(max_examples):
        if mode == "atheris":
            # disable our blackbox for tests
            mutated = custom_mutator(start, random=r, blackbox=False)
        if mode == "baseline":
            mutated = r.randbytes(BUFFER_SIZE)
        fuzz_one_input(mutated)


@pytest.mark.parametrize(
    "strategy",
    [
        # string
        st.text(),
        st.text(min_size=0, max_size=0),
        st.text(min_size=10),
        st.text(min_size=10, max_size=15),
        # integer
        st.integers(),
        st.integers(-100),
        st.integers(100),
        st.integers(0, 0),
        st.integers(0, 5),
        st.integers(-5, 0),
        st.integers(-(2**64), 2**32),
        # bytes
        st.binary(),
        st.binary(min_size=0, max_size=0),
        st.binary(min_size=10),
        st.binary(min_size=5, max_size=10),
        # float
        st.floats(),
        st.floats(min_value=-10),
        st.floats(max_value=10),
        st.floats(-10, 10),
        st.floats(allow_nan=False),
        st.floats(allow_infinity=False),
        # bool
        st.booleans(),
        # composite / weird / other
        st.just(None),
        st.lists(st.integers()),
        st.lists(st.floats()),
        st.lists(st.booleans()),
        st.lists(st.text()),
        st.lists(st.binary()),
    ],
    ids=get_pretty_function_description,
)
@given(start=st.binary())
@settings(deadline=None, max_examples=5)
def test_runs_with_various_kwargs(start, strategy):
    @given(strategy)
    def f(x):
        pass

    fuzz(f, start=start, mode="atheris", max_examples=100)


@example(-198237154, 1928391283)
@example(50_000, 100_000)
@given(st.integers(), st.integers())
@settings(deadline=None, max_examples=10, database=None)
def test_can_find_endpoints(min_value, max_value):
    assume(min_value <= max_value)

    @run_to_buffer
    def start(data):
        data.draw(st.integers(min_value, max_value))
        data.mark_interesting()

    for target in [min_value, max_value]:
        print("target", target)

        @settings(database=None)
        @given(st.integers(min_value, max_value))
        def f(n):
            assert n != target, MARKER

        with pytest.raises(AssertionError, match=MARKER):
            print("atheris")
            fuzz(f, start=start, mode="atheris", max_examples=1_000)

        with pytest.raises(AssertionError, match=MARKER):
            print("baseline")
            fuzz(f, start=start, mode="baseline", max_examples=1_000)


@pytest.mark.parametrize(
    "target, offset",
    [(192837123, 8), (-8712313, 4), (918273, 1), (-94823, -1), (912873192312387, -5)],
)
@pytest.mark.parametrize(
    "min_offset, max_offset", [(None, None), (1234, None), (None, 1234), (1234, 1234)]
)
@flaky(max_runs=3, min_passes=1)
def test_can_find_nearby_integers(target, offset, min_offset, max_offset):
    min_value = None if min_offset is None else target - min_offset
    max_value = None if max_offset is None else target + max_offset

    @settings(database=None)
    @given(st.integers(min_value=min_value, max_value=max_value))
    def f(n):
        assert n != target + offset, MARKER

    @run_to_buffer
    def start(data):
        # remove this when we move st.integers weighting off of two-integer-draws
        if min_offset is not None and max_offset is not None:
            data.draw_integer(0, 127, forced=20)
        data.draw_integer(min_value=min_value, max_value=max_value, forced=target)
        data.mark_interesting()

    # atheris should find this and baseline shouldn't.
    with pytest.raises(AssertionError, match=MARKER):
        fuzz(f, start=start, mode="atheris", max_examples=1_000)

    fuzz(f, start=start, mode="baseline", max_examples=1_000)


@pytest.mark.parametrize(
    "target, offset",
    [(192837123, 8), (-8712313, 4), (918273, 2), (-94823, -2), (912873192312387, -5)],
)
@pytest.mark.parametrize(
    "min_offset, max_offset", [(None, None), (1234, None), (None, 1234), (1234, 1234)]
)
@flaky(max_runs=3, min_passes=1)
def test_can_find_nearby_floats(target, offset, min_offset, max_offset):
    min_value = next_up(-math.inf) if min_offset is None else float(target - min_offset)
    max_value = (
        next_down(math.inf) if max_offset is None else float(target + max_offset)
    )

    @settings(database=None)
    @given(
        st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    def f(n):
        # we're never going to hit the offset exactly with floats. just get close.
        assert abs(n - (target + offset)) > 1.5, MARKER

    @run_to_buffer
    def start(data):
        data.draw_float(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            smallest_nonzero_magnitude=SMALLEST_SUBNORMAL,
            forced=target,
        )
        data.mark_interesting()

    with pytest.raises(AssertionError, match=MARKER):
        fuzz(f, start=start, mode="atheris", max_examples=1_000)

    fuzz(f, start=start, mode="baseline", max_examples=1_000)


@pytest.mark.parametrize(
    "target",
    # copying another interval
    ["xXaaxX", "bbxXxX", "axXaxX", "bbaaaa"] +
    # copying two intervals (TODO this is very flaky)
    # ["xXxXxX"] +
    # deleting an interval
    ["bbaax", "bbaa", "bba", "bb", "b", "", "bbX", "axX"],
    # TODO tests for inserting a new value / replacing an interval with a new value
)
@flaky(max_runs=3, min_passes=1)
def test_can_splice_strings(target):
    @settings(database=None)
    @given(st.text())
    def f(s):
        assert s != target, MARKER

    @run_to_buffer
    def start(data):
        data.draw_string(
            intervals=IntervalSet(((0, 55295), (57344, 1114111))),
            min_size=0,
            max_size=math.inf,
            forced="bbaaxX",
        )
        data.mark_interesting()

    with pytest.raises(AssertionError, match=MARKER):
        fuzz(f, start=start, mode="atheris", max_examples=10_000)

    # hypothesis does a good job of finding small strings in the ascii range.
    if len(target) > 1:
        fuzz(f, start=start, mode="baseline", max_examples=10_000)


@pytest.mark.parametrize(
    "target",
    # copying another interval
    [b"xXaaxX", b"bbxXxX", b"axXaxX", b"bbaaaa"] +
    # copying two intervals (TODO this is very flaky)
    # [b"xXxXxX"] +
    # deleting an interval
    [b"bbaax", b"bbaa", b"bba", b"bb", b"b", b"", b"bbX", b"axX"],
    # TODO tests for inserting a new value / replacing an interval with a new value
)
@flaky(max_runs=3, min_passes=1)
def test_can_splice_bytes(target):
    @settings(database=None)
    @given(st.binary())
    def f(s):
        assert s != target, MARKER

    @run_to_buffer
    def start(data):
        data.draw_bytes(
            min_size=0,
            max_size=math.inf,
            forced=b"bbaaxX",
        )
        data.mark_interesting()

    with pytest.raises(AssertionError, match=MARKER):
        fuzz(f, start=start, mode="atheris", max_examples=10_000)

    if len(target) > 1:
        fuzz(f, start=start, mode="baseline", max_examples=10_000)

@example(IntervalSet.from_string("abcdefg"))
@example(IntervalSet.from_string(string.printable))
@example(IntervalSet.from_string(string.ascii_lowercase))
@example(IntervalSet.from_string(string.hexdigits))
@example(IntervalSet.from_string(string.printable + chr(1000)))
@example(IntervalSet(((0, 254),)))
@example(IntervalSet(((0, 255),)))
@example(IntervalSet(((0, 256),)))
@example(IntervalSet(((0, 257),)))
@example(IntervalSet(((1000, 1000),)))
@example(IntervalSet(((0, 20), (40, 60))))
@given(intervals=intervals(min_size=1))
def test_mutate_string_on_weird_intervals(intervals):
    # please dont crash on weird bounds
    for _ in range(100):
        mutate_string(
            ".........",
            min_size=0,
            max_size=100,
            intervals=intervals,
            random=r,
        )
