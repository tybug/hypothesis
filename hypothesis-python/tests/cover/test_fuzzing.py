from hypothesis import settings, given, assume, example
from hypothesis import strategies as st
from tests.conjecture.common import run_to_buffer
from hypothesis.core import custom_mutator, BUFFER_SIZE
import random
from random import Random
import pytest
import uuid

from tests.common.utils import flaky

MARKER = uuid.uuid4().hex


def fuzz(f, *, start, mode, max_examples):
    fuzz_one_input = f.hypothesis._get_fuzz_target(
        args=(), kwargs={}, use_atheris=mode == "atheris"
    )
    fuzz_one_input(start)
    for _ in range(max_examples):
        if mode == "atheris":
            r = Random(random.randint(0, int(1e10)))
            # avoid our blackbox for num_calls < 100
            mutated = custom_mutator(start, random=r, num_calls=100)
        if mode == "baseline":
            mutated = random.randbytes(BUFFER_SIZE)
        fuzz_one_input(mutated)


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
@flaky(max_runs=5, min_passes=1)
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
    with pytest.raises(AssertionError):
        fuzz(f, start=start, mode="atheris", max_examples=2_000)

    fuzz(f, start=start, mode="baseline", max_examples=2_000)
