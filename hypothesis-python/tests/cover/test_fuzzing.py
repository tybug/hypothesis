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
from hypothesis.core import BUFFER_SIZE
from hypothesis.database import ir_to_bytes
from hypothesis.fuzzing import (
    AtherisProvider,
    Draw,
    NodeMutator,
    CollectionMutator,
    custom_mutator,
    mutate_string,
)
from hypothesis.internal.conjecture.data import ConjectureData, ir_value_equal
from hypothesis.internal.floats import next_down, next_up
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.internal.reflection import get_pretty_function_description

from tests.common.strategies import intervals
from tests.common.utils import flaky
from tests.conjecture.common import draw_value, ir_types_and_kwargs

MARKER = uuid.uuid4().hex

# stop hypothesis from seeding our random
r = Random(random.randint(0, int(1e10)))


@st.composite
def serialized_ir(draw):
    values = draw(
        st.lists(
            st.booleans()
            | st.integers()
            | st.floats()
            | st.text(st.characters())
            | st.binary()
        )
    )
    return ir_to_bytes(values)


@st.composite
def draws(draw):
    ir_type, kwargs = draw(ir_types_and_kwargs())
    value = draw_value(ir_type, kwargs)
    return Draw(ir_type=ir_type, kwargs=kwargs, value=value, forced=None)


def fuzz(f, *, start, mode, max_examples):
    fuzz_one_input = f.hypothesis._get_fuzz_target(
        args=(), kwargs={}, use_atheris=mode == "atheris"
    )
    if isinstance(start, list):
        start = ir_to_bytes(start)
    elif not isinstance(start, bytes):
        assert False, "must be either a bytes-serialized ir or list of ir nodes"
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
@given(start=st.binary() | serialized_ir())
@settings(deadline=None, max_examples=5)
def test_runs_with_various_kwargs(start, strategy):
    @given(strategy)
    def f(x):
        pass

    fuzz(f, start=start, mode="atheris", max_examples=100)


# I'm pretty nervous that we'll violate a forced invariant and return an unsound value.
# this is a start at sanity checking that
@given(st.data(), serialized_ir() | st.binary())
@settings(max_examples=20, deadline=None)
def test_respects_list_size(data, start):
    min_size = data.draw(st.integers(0, 10))
    max_size = data.draw(st.integers(min_size, min_size + 10))

    @given(st.lists(st.integers(), min_size=min_size, max_size=max_size))
    def f(l):
        assert min_size <= len(l) <= max_size

    fuzz(f, start=start, mode="atheris", max_examples=100)


@example(-198237154, 1928391283)
@example(50_000, 100_000)
@given(st.integers(), st.integers())
@settings(deadline=None, max_examples=10, database=None)
def test_can_find_endpoints(min_value, max_value):
    assume(min_value <= max_value)

    start = []
    if max_value - min_value > 127:
        start.append(42)
    start.append((min_value + max_value) // 2)

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

    start = []
    # remove this when we move st.integers weighting off of two-integer-draws
    if min_offset is not None and max_offset is not None:
        start.append(42)
    start.append(target)

    # atheris should find this and baseline shouldn't.
    with pytest.raises(AssertionError, match=MARKER):
        fuzz(f, start=start, mode="atheris", max_examples=1_000)

    fuzz(f, start=start, mode="baseline", max_examples=1_000)


@pytest.mark.parametrize(
    "target, offset",
    [
        (192837123.0, 8.0),
        (-8712313.0, 4.0),
        (918273.0, 2.0),
        (-94823.0, -2.0),
        (912873192312387.0, -5.0),
    ],
)
@pytest.mark.parametrize(
    "min_offset, max_offset",
    [(None, None), (1234.0, None), (None, 1234.0), (1234.0, 1234.0)],
)
@flaky(max_runs=3, min_passes=1)
def test_can_find_nearby_floats(target, offset, min_offset, max_offset):
    min_value = next_up(-math.inf) if min_offset is None else target - min_offset
    max_value = next_down(math.inf) if max_offset is None else target + max_offset

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

    start = [target]
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

    start = ["bbaaxX"]
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

    start = [b"bbaaxX"]
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


@given(
    # cover the case where basically nothing aligns with st.binary(). serialized_ir()
    # will align if we get lucky and the draws from ir_types_and_kwargs() line
    # up, and will produce misalignments (draw a random value) otherwise. all of
    # these combinations are good for testing since the nondeterminism comes into
    # play largely or entirely when we draw new random values.
    serialized_ir() | st.binary(),
    st.lists(ir_types_and_kwargs()),
)
@settings(max_examples=500)
def test_replaying_buffer_is_deterministic(buffer, draws):
    data1 = ConjectureData(BUFFER_SIZE, b"", provider=AtherisProvider)
    data2 = ConjectureData(BUFFER_SIZE, b"", provider=AtherisProvider)
    data1.provider.buffer = buffer
    data2.provider.buffer = buffer

    # we're currently abusing the context manager for init purposes on the provider
    with (
        data1.provider.per_test_case_context_manager(),
        data2.provider.per_test_case_context_manager(),
    ):
        for ir_type, kwargs in draws:
            v1 = getattr(data1, f"draw_{ir_type}")(**kwargs)
            v2 = getattr(data2, f"draw_{ir_type}")(**kwargs)
            assert ir_value_equal(ir_type, v1, v2)


@pytest.mark.parametrize(
    "strat1, strat2, start, eq",
    [
        (
            s := [
                st.integers(),
                st.integers(min_value=1, max_value=2),
                st.lists(st.booleans()),
            ],
            s,
            # s1 = (42**3, 2, [False])
            # s2 = (100, 1, [])
            [42**3, 2, True, False, False] + [100, 1, False],
            lambda v1, v2: v1 == v2,
        ),
        (
            [st.integers(), st.binary()] + [st.booleans()],
            [st.integers(), st.binary()] + [st.floats()],
            [111111, b"aaaaaa", True] + [222222, b"bbbbbb", 0.1],
            lambda v1, v2: v1[0:2] == v2[0:2],
        ),
        (
            [st.booleans()] + [st.integers(), st.binary()],
            [st.floats()] + [st.integers(), st.binary()],
            [True, 111111, b"aaaaaa"] + [0.1, 222222, b"bbbbbb"],
            lambda v1, v2: v1[1:3] == v2[1:3],
        ),
        (
            [st.booleans()] + [st.integers(), st.binary()],
            [st.integers(), st.binary()] + [st.floats()],
            [True, 111111, b"aaaaaa"] + [222222, b"bbbbbb", 0.1],
            lambda v1, v2: v1[1:3] == v2[0:2],
        ),
        (
            [st.booleans(), st.integers(), st.binary(), st.floats()],
            [st.floats(), st.integers(), st.binary(), st.booleans()],
            [True, 111111, b"aaaaaa", 0.1] + [0.1, 222222, b"bbbbbb", True],
            lambda v1, v2: v1[1:3] == v2[1:3],
        ),
    ],
)
def test_can_copy_sequences_of_nodes(strat1, strat2, start, eq):
    s1 = st.composite(lambda draw: [draw(s) for s in strat1])
    s2 = st.composite(lambda draw: [draw(s) for s in strat2])

    print(strat1, strat2, start, eq)

    @settings(database=None)
    @given(s1(), s2())
    def f(v1, v2):
        assert not eq(v1, v2), MARKER

    with pytest.raises(AssertionError, match=MARKER):
        print("mode: atheris")
        fuzz(f, start=start, mode="atheris", max_examples=200)

    print("mode: baseline")
    fuzz(f, start=start, mode="baseline", max_examples=1000)


@given(st.lists(draws(), min_size=1), st.integers(1, 100))
@settings(max_examples=200)
def test_mutator_with_forced_nodes(draws, total_cost):
    print(f"draws: {draws}")
    # I would like to make a random number of nodes forced and then check that their
    # value didn't change, but I don't know how to track a specific node throughout
    # all the mutations to its final destination.
    draw = r.choice(draws)
    forced_value = draw.value
    draw.forced = forced_value

    mutated_draws = NodeMutator(total_cost=total_cost, draws=draws, random=r).mutate()
    for draw in mutated_draws:
        if draw.forced is None:
            continue
        assert ir_value_equal(draw.ir_type, draw.value, forced_value)


def test_collection_mutator_mutates_empty_collection():
    def draw_element():
        return r.choice("abcdefgh")

    empties = 0
    for _ in range(100):
        mutated = "".join(
            CollectionMutator(
                value=[],
                min_size=0,
                max_size=100,
                draw_element=draw_element,
                random=r,
            ).mutate()
        )
        if mutated == "":
            empties += 1

    # it's possible to mutate such that we insert a val and then immediate
    # delete it. but this shouldn't happen very often.
    assert empties <= 10, empties


@given(st.text(), st.integers(0, 10), st.integers(0, 20))
@settings(max_examples=1000)
def test_collection_mutator(s, min_size, max_size):
    assume(min_size <= max_size)

    def draw_element():
        return r.choice("abcdefg")

    "".join(
        CollectionMutator(
            value=list(s),
            min_size=min_size,
            max_size=max_size,
            draw_element=draw_element,
            random=r,
        ).mutate()
    )


@given(st.lists(draws()))
def test_aligned_provider(draws):
    data = ConjectureData(BUFFER_SIZE, b"", provider=AtherisProvider)
    data.provider.buffer = ir_to_bytes([d.value for d in draws])

    with data.provider.per_test_case_context_manager():
        for draw in draws:
            v = getattr(data, f"draw_{draw.ir_type}")(**draw.kwargs)
            assert ir_value_equal(draw.ir_type, v, draw.value)
