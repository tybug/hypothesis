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
from contextlib import contextmanager

from hypothesis import HealthCheck, assume, settings, strategies as st
from hypothesis.control import current_build_context
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture import engine as engine_module
from hypothesis.internal.conjecture.data import ConjectureData, IRNode, Status
from hypothesis.internal.conjecture.engine import BUFFER_SIZE, ConjectureRunner
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.entropy import deterministic_PRNG
from hypothesis.internal.floats import SMALLEST_SUBNORMAL, sign_aware_lte
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.strategies._internal.strings import OneCharStringStrategy, TextStrategy

from tests.common.strategies import intervals

SOME_LABEL = calc_label_from_name("some label")


TEST_SETTINGS = settings(
    max_examples=5000, database=None, suppress_health_check=list(HealthCheck)
)


def run_to_data(f):
    with deterministic_PRNG():
        runner = ConjectureRunner(f, settings=TEST_SETTINGS)
        runner.run()
        assert runner.interesting_examples
        (last_data,) = runner.interesting_examples.values()
        return last_data


def run_to_buffer(f):
    return bytes(run_to_data(f).buffer)


@contextmanager
def buffer_size_limit(n):
    original = engine_module.BUFFER_SIZE
    try:
        engine_module.BUFFER_SIZE = n
        yield
    finally:
        engine_module.BUFFER_SIZE = original


def shrinking_from(start):
    def accept(f):
        with deterministic_PRNG():
            runner = ConjectureRunner(
                f,
                settings=settings(
                    max_examples=5000,
                    database=None,
                    suppress_health_check=list(HealthCheck),
                ),
            )
            runner.cached_test_function(start)
            assert runner.interesting_examples
            (last_data,) = runner.interesting_examples.values()
            return runner.new_shrinker(
                last_data, lambda d: d.status == Status.INTERESTING
            )

    return accept


def fresh_data(*, random=None, observer=None) -> ConjectureData:
    if random is None:
        try:
            context = current_build_context()
        except InvalidArgument:
            # ensure usage of fresh_data() is not flaky outside of property tests.
            raise ValueError(
                "must pass a seeded Random instance to fresh_data() when "
                "outside of a build context"
            ) from None

        # within property tests, ensure fresh_data uses a controlled source of
        # randomness.
        # drawing this from the current build context is almost *too* magical. But
        # the alternative is an extra @given(st.randoms()) everywhere we use
        # fresh_data, so eh.
        random = context.data.draw(st.randoms())

    return ConjectureData(
        BUFFER_SIZE,
        prefix=b"",
        random=random,
        observer=observer,
    )


@st.composite
def draw_integer_kwargs(
    draw,
    *,
    use_min_value=True,
    use_max_value=True,
    use_shrink_towards=True,
    use_weights=True,
    use_forced=False,
):
    min_value = None
    max_value = None
    shrink_towards = 0
    weights = None

    forced = draw(st.integers()) if use_forced else None
    # handle the weights case entirely independently from the non-weights case.
    if use_weights:
        assert use_max_value
        assert use_min_value

        min_value = draw(st.integers(max_value=forced))
        min_val = max(min_value, forced) if forced is not None else min_value
        max_value = draw(st.integers(min_value=min_val))

        weights = {}
        # totally partition [min_value, max_value] into intervals. Treat partition
        # list [a, b, c, d] as intervals [(a, b), (b + 1, c), (c + 1, d)].
        # TODO this doesn't cover the case of multiple intervals in a single
        # IntervalSet.
        partitions = draw(
            st.lists(
                st.integers(min_value, max_value), unique=True, max_size=255 - 1
            ).map(sorted)
        )
        # avoid overlapping endpoints. eg [(1, 1), (1, 2)] is invalid.
        if min_value not in partitions:
            partitions = [min_value] + partitions
        if max_value not in partitions:
            partitions = partitions + [max_value]

        # Sampler doesn't play well with super small floats, so exclude them
        nonzero_probs = st.floats(0.01, 1)
        for i, (a, b) in enumerate(zip(partitions, partitions[1:])):
            if i > 0:
                a += 1
            if forced is not None and a <= forced <= b:
                # forced's interval has to have nonzero probability, because it must
                # be possible to generate forced values
                p = draw(nonzero_probs)
            else:
                # otherwise, this interval can have 0 probability
                p = draw(st.one_of(st.just(0), nonzero_probs))
            weights[IntervalSet([(a, b)])] = p

        # re-normalize probabilities to sum to 1
        total = sum(weights.values())
        # invalid to have a weighting that disallows all possibilities
        assume(total != 0)
        weights = {k: v / total for k, v in weights.items()}
        # float rounding error can cause the sum to fail a similar internal
        # invariant here. This isn't an issue in practice because weights are an
        # internal api and we use sane p values.
        assume(sum(weights.values()) == 1)
    else:
        if use_min_value:
            min_value = draw(st.integers(max_value=forced))
        if use_max_value:
            min_vals = []
            if min_value is not None:
                min_vals.append(min_value)
            if forced is not None:
                min_vals.append(forced)
            min_val = max(min_vals) if min_vals else None
            max_value = draw(st.integers(min_value=min_val))

    if use_shrink_towards:
        shrink_towards = draw(st.integers())

    if forced is not None:
        assume((forced - shrink_towards).bit_length() < 128)

    return {
        "min_value": min_value,
        "max_value": max_value,
        "shrink_towards": shrink_towards,
        "weights": weights,
        "forced": forced,
    }


@st.composite
def draw_string_kwargs(draw, *, use_min_size=True, use_max_size=True, use_forced=False):
    # TODO also sample empty intervals, ie remove this min_size, once we handle empty
    # pseudo-choices in the ir
    interval_set = draw(intervals(min_size=1))
    forced = (
        draw(TextStrategy(OneCharStringStrategy(interval_set))) if use_forced else None
    )

    min_size = 0
    max_size = None

    if use_min_size:
        # cap to some reasonable min size to avoid overruns.
        n = 100
        if forced is not None:
            n = min(n, len(forced))

        min_size = draw(st.integers(0, n))

    if use_max_size:
        n = min_size if forced is None else max(min_size, len(forced))
        max_size = draw(st.integers(min_value=n))
        # cap to some reasonable max size to avoid overruns.
        max_size = min(max_size, min_size + 100)

    return {
        "intervals": interval_set,
        "min_size": min_size,
        "max_size": max_size,
        "forced": forced,
    }


@st.composite
def draw_bytes_kwargs(draw, *, use_forced=False):
    forced = draw(st.binary()) if use_forced else None
    # be reasonable with the number of bytes we ask for. We only have BUFFER_SIZE
    # to work with before we overrun.
    size = (
        draw(st.integers(min_value=0, max_value=100)) if forced is None else len(forced)
    )

    return {"size": size, "forced": forced}


@st.composite
def draw_float_kwargs(
    draw, *, use_min_value=True, use_max_value=True, use_forced=False
):
    forced = draw(st.floats()) if use_forced else None
    pivot = forced if (use_forced and not math.isnan(forced)) else None
    min_value = -math.inf
    max_value = math.inf
    smallest_nonzero_magnitude = SMALLEST_SUBNORMAL
    allow_nan = True if (use_forced and math.isnan(forced)) else draw(st.booleans())

    if use_min_value:
        min_value = draw(st.floats(max_value=pivot, allow_nan=False))

    if use_max_value:
        if pivot is None:
            min_val = min_value
        else:
            min_val = pivot if sign_aware_lte(min_value, pivot) else min_value
        max_value = draw(st.floats(min_value=min_val, allow_nan=False))

    largest_magnitude = max(abs(min_value), abs(max_value))
    # can't force something smaller than our smallest magnitude.
    if pivot is not None and pivot != 0.0:
        largest_magnitude = min(largest_magnitude, pivot)

    # avoid drawing from an empty range
    if largest_magnitude > 0:
        smallest_nonzero_magnitude = draw(
            st.floats(
                min_value=0,
                # smallest_nonzero_magnitude breaks internal clamper invariants if
                # it is allowed to be larger than the magnitude of {min, max}_value.
                max_value=largest_magnitude,
                allow_nan=False,
                exclude_min=True,
                allow_infinity=False,
            )
        )
    return {
        "min_value": min_value,
        "max_value": max_value,
        "forced": forced,
        "allow_nan": allow_nan,
        "smallest_nonzero_magnitude": smallest_nonzero_magnitude,
    }


@st.composite
def draw_boolean_kwargs(draw, *, use_forced=False):
    forced = draw(st.booleans()) if use_forced else None
    p = draw(st.floats(0, 1, allow_nan=False, allow_infinity=False))

    # avoid invalid forced combinations
    assume(p > 0 or forced is False)
    assume(p < 1 or forced is True)

    if 0 < p < 1:
        # match internal assumption about avoiding large draws
        bits = math.ceil(-math.log(min(p, 1 - p), 2))
        assume(bits <= 64)

    return {"p": p, "forced": forced}


def kwargs_strategy(ir_type):
    return {
        "boolean": draw_boolean_kwargs(),
        "integer": draw_integer_kwargs(),
        "float": draw_float_kwargs(),
        "bytes": draw_bytes_kwargs(),
        "string": draw_string_kwargs(),
    }[ir_type]


def ir_types_and_kwargs():
    options = ["boolean", "integer", "float", "bytes", "string"]
    return st.one_of(
        st.tuples(st.just(name), kwargs_strategy(name)) for name in options
    )


def draw_value(ir_type, kwargs):
    data = fresh_data()
    return getattr(data, f"draw_{ir_type}")(**kwargs)


@st.composite
def ir_nodes(draw, *, was_forced=None, ir_type=None):
    if ir_type is None:
        (ir_type, kwargs) = draw(ir_types_and_kwargs())
    else:
        kwargs = draw(kwargs_strategy(ir_type))
    # ir nodes don't include forced in their kwargs. see was_forced attribute
    del kwargs["forced"]
    value = draw_value(ir_type, kwargs)
    was_forced = draw(st.booleans()) if was_forced is None else was_forced

    return IRNode(ir_type=ir_type, value=value, kwargs=kwargs, was_forced=was_forced)
