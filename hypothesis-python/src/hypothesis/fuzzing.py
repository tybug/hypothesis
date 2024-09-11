# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""This module provides the core primitives of Hypothesis, such as given."""

import math
import time
from contextlib import contextmanager
from random import Random
from typing import (
    Mapping,
    Optional,
    Tuple,
)


from hypothesis.errors import (
    StopTest,
)
from hypothesis.internal import floats as flt
from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.compat import (
    int_from_bytes,
    int_to_bytes,
)
from hypothesis.internal.conjecture.data import (
    BYTE_MASKS,
    COLLECTION_DEFAULT_MAX_SIZE,
    NASTY_FLOATS,
    ConjectureData,
    IRKWargsType,
    IRType,
    IRTypeName,
    Status,
    bits_to_bytes,
    ir_value_permitted,
)
from hypothesis.internal.conjecture.engine import (
    BUFFER_SIZE,
    PrimitiveProvider,
)
from hypothesis.internal.conjecture.junkdrawer import (
    clamp,
    replace_all,
)
from hypothesis.internal.conjecture.utils import _calc_p_continue
from hypothesis.internal.floats import next_down, next_up, sign_aware_lte



BOUNDS_CACHE_SIZE = 16_384  # 2**14
INT_SIZES = (8, 16, 32, 64, 128)
INT_SIZES_WEIGHTS = (4.0, 8.0, 1.0, 1.0, 0.5)
FLOAT_SIZES = (8, 16, 32, 64, 128, 1024)
FLOAT_SIZES_WEIGHTS = (4.0, 8.0, 1.0, 1.0, 0.5, 0.5)
data_to_bounds: Mapping[
    bytes, Mapping[Tuple[int, int], Tuple[IRTypeName, IRKWargsType, IRType]]
] = LRUReusedCache(BOUNDS_CACHE_SIZE)

statistics = {
    "per_item_stats": [],
    "num_calls": 0,
    "time_mutating": 0,
}
track_per_item_stats = False
custom_mutator_called = False
print_stats_at = 25_000
stats_printed = True  # set to True to disable printing entirely


def _size(*, min_size, max_size, average_size, random):
    p_continue = _calc_p_continue(average_size - min_size, max_size - min_size)
    size = min_size
    while random.random() < p_continue and size < max_size:
        size += 1
    return size


def num_mutations(*, min_size, max_size, random):
    # TODO tweak this distribution?
    average_size = min(
        max(
            min_size + 3,
            # for targets with a large amount of nodes, mutate 10% of them on average.
            # otherwise we would mutate basically nothing for larger nodes
            0.1 * (min_size + max_size),
        ),
        0.5 * (min_size + max_size),
    )
    return _size(
        min_size=min_size, max_size=max_size, average_size=average_size, random=random
    )


def random_float_between(min_value, max_value, smallest_nonzero_magnitude, *, random):
    assert not math.isinf(min_value)
    assert not math.isinf(max_value)

    r = random.random()
    if flt.is_negative(min_value):
        if flt.is_negative(max_value):
            max_point = min(max_value, -smallest_nonzero_magnitude)
            f = max_point + (min_value - max_point) * (1 - r)
        else:
            start = min_value
            # similar to libfuzzer, ref
            # https://github.com/llvm/llvm-project/blob/e4f3b56dae25e792b4aa5b009e371c8836fdc2fa/
            # compiler-rt/include/fuzzer/FuzzedDataProvider.h#L248
            if math.isinf(max_value - min_value):
                # max_value - min_value can overflow. use half of the diff as the
                # range and use the first and second half of the diff with equal prob.
                total_range = (max_value / 2.0) - (min_value / 2.0)
                if random.randint(0, 1) == 0:
                    start += total_range
            else:
                total_range = max_value - min_value
            f = start + total_range * r

            # I don't really want to deal with this case right now because it splits
            # the range into two segments and I have to bookkeep the relative sizes etc.
            # this should be a very rare case, so effectively throw it away if it happens.
            if abs(f) < smallest_nonzero_magnitude and f != 0:
                f = min_value
    else:
        # case: both positive.
        min_point = max(min_value, smallest_nonzero_magnitude)
        f = min_point + (max_value - min_point) * r
    return f


def _make_serializable(ir_value):
    # this isn't free, so we should only run this when we're explicitly tracking stats.
    assert track_per_item_stats
    if type(ir_value) in [bytes, bytearray]:
        return str(ir_value)
    return ir_value


def _mutate_integer(*, min_value, max_value, random):
    # roughly equivalent to shrink_towards in draw_integer, but without
    # shrinking semantics.
    origin = 0
    if min_value is not None:
        origin = max(min_value, origin)
    if max_value is not None:
        origin = min(max_value, origin)

    def _unbounded_integer():
        # bias towards smaller values. distribution copied from draw_integer
        bits = random.choices(INT_SIZES, INT_SIZES_WEIGHTS, k=1)[0]
        radius = 2 ** (bits - 1) - 1
        return random.randint(-radius, radius)

    if min_value is None and max_value is None:
        forced = _unbounded_integer()
    elif min_value is None:
        assert max_value is not None
        forced = max_value + 1
        while forced > max_value:
            forced = origin + _unbounded_integer()
    elif max_value is None:
        assert min_value is not None
        forced = min_value - 1
        while forced < min_value:
            forced = origin + _unbounded_integer()
    else:
        assert min_value is not None
        assert max_value is not None
        # somewhat hodgepodge amalgamation of what hypothesis does for
        # weighting the size of bounded ints, and what I think works
        # better for fuzzing (e.g. lowering the bit limit, decreasing the
        # probability from 7/8 to 1/2).
        bits = (max_value - min_value).bit_length()
        if bits > 18 and random.randint(0, 1) == 0:
            bits = min(bits, random.choices(INT_SIZES, INT_SIZES_WEIGHTS, k=1)[0])
            radius = 2 ** (bits - 1) - 1
            forced = origin + random.randint(-radius, radius)
            forced = clamp(min_value, forced, max_value)
        else:
            forced = random.randint(min_value, max_value)

        # right now we get integer endpoints for free because hypothesis draws
        # two integers for st.integers, one of which controls this endpoint.
        # once we roll that into a single call we'll need to introduce this.
        # if (max_value - min_value > 300) and (r := random.randint(0, 100)) < 4:
        #     forced = {
        #         0: min_value,
        #         1: min_value + 1,
        #         2: max_value - 1,
        #         3: max_value
        #     }[r]
    return forced


def nearby_number(value, *, min_value, max_value, random_value):
    r = 10
    min_point = value - r
    max_point = value + r
    if min_value is not None:
        min_point = max(min_value, min_point)
    if max_value is not None:
        max_point = min(max_value, max_point)

    forced = random_value(min_point, max_point)
    assert min_value is None or min_value <= forced
    assert max_value is None or forced <= max_value
    return forced


def mutate_integer(value, *, min_value, max_value, random):
    # with some probability, draw in a small area around the previous value.
    if (
        not (
            min_value is not None
            and max_value is not None
            and (max_value - min_value) < 300
        )
        and random.randint(0, 10) == 0
    ):

        def random_value(min_point, max_point):
            return random.randint(int(min_point), int(max_point))

        forced = nearby_number(
            value,
            min_value=min_value,
            max_value=max_value,
            random_value=random_value,
        )
    else:
        forced = _mutate_integer(
            min_value=min_value, max_value=max_value, random=random
        )
    return forced


def mutate_float(
    value, *, min_value, max_value, allow_nan, smallest_nonzero_magnitude, random
):
    # with some probability, draw in a small area around the previous value.
    if (
        (max_value - min_value) >= 300
        and not math.isinf(value)
        and not math.isnan(value)
        and random.randint(0, 10) == 0
    ):

        def random_value(min_point, max_point):
            return random_float_between(
                min_point, max_point, smallest_nonzero_magnitude, random=random
            )

        forced = nearby_number(
            value,
            min_value=min_value,
            max_value=max_value,
            random_value=random_value,
        )
    else:
        forced = _mutate_float(
            min_value=min_value,
            max_value=max_value,
            allow_nan=allow_nan,
            smallest_nonzero_magnitude=smallest_nonzero_magnitude,
            random=random,
        )
    return forced


def _mutate_float(
    *, min_value, max_value, allow_nan, smallest_nonzero_magnitude, random
):
    def is_inf(value, *, sign):
        return math.copysign(1.0, value) == sign and math.isinf(value)

    def permitted(f):
        if math.isnan(f):
            return allow_nan
        if 0 < abs(f) < smallest_nonzero_magnitude:
            return False
        return sign_aware_lte(min_value, f) and sign_aware_lte(f, max_value)

    # draw a "nasty" value with probability 0.05. I think hypothesis uses 0.2,
    # but they have a significantly smaller budget and also count duplicates,
    # so we should have a lower p.
    boundary_values = [
        min_value,
        next_up(min_value),
        min_value + 1,
        max_value - 1,
        next_down(max_value),
        max_value,
    ]
    nasty_floats = [f for f in NASTY_FLOATS + boundary_values if permitted(f)]
    if random.randint(0, 99) < 5 and nasty_floats:
        forced = random.choice(nasty_floats)
    else:
        min_val = min_value
        max_val = max_value
        # we already generating inf via nasty_floats. constrain to real
        # floats here.
        if is_inf(min_value, sign=-1.0):
            min_val = next_up(min_value)
        if is_inf(min_value, sign=1.0):
            min_val = next_down(min_value)
        if is_inf(max_value, sign=-1.0):
            max_val = next_up(max_value)
        if is_inf(max_value, sign=1.0):
            max_val = next_down(max_value)

        assert not math.isinf(min_val)
        assert not math.isinf(max_val)
        assert sign_aware_lte(min_val, max_val)

        origin = 0
        if min_value is not None:
            origin = max(min_value, origin)
        if max_value is not None:
            origin = min(max_value, origin)

        # weight towards smaller floats - like we do for ints, but even
        # more heavily, as the range can be enormous.
        diff = max_val - min_val
        # max - min can overflow to inf at the float boundary.
        bits = int(diff).bit_length() if not math.isinf(diff) else 1024
        if bits > 18 and random.randint(0, 7) < 7:
            bits = min(bits, random.choices(FLOAT_SIZES, FLOAT_SIZES_WEIGHTS, k=1)[0])
            radius = float(2 ** (bits - 1) - 1)
            forced = origin + random_float_between(
                -radius, radius, smallest_nonzero_magnitude, random=random
            )
            forced = clamp(min_value, forced, max_value)
        else:
            forced = random_float_between(
                min_val, max_val, smallest_nonzero_magnitude, random=random
            )
    # with probability 0.05, truncate to an integer-valued float
    if (
        random.randint(0, 99) < 5
        and not math.isnan(forced)
        and not math.isinf(forced)
        and permitted(truncated := float(math.floor(forced)))
    ):
        forced = truncated

    return forced


def mutate_collection(value, *, min_size, max_size, draw_element, random):
    if max_size is None:
        max_size = COLLECTION_DEFAULT_MAX_SIZE

    value = list(value)

    def draw_value(*, min_size, max_size, average_size):
        size = _size(
            min_size=min_size,
            max_size=max_size,
            average_size=average_size,
            random=random,
        )
        return [draw_element() for _ in range(size)]

    # totally random with probability 0.1, more intelligent mutation
    # otherwise. This is me being scared of not being able to get out of small
    # size collections like the empty string, because there's nothing to mutate.
    if random.randint(0, 9) == 0:
        # copied from HypothesisProvider.draw_string
        average_size = min(
            max(min_size * 2, min_size + 5),
            0.5 * (min_size + max_size),
        )
        forced = draw_value(
            min_size=min_size, max_size=max_size, average_size=average_size
        )
    else:
        # pick n splice points. for each [n1, n2] interval, do one of:
        # * delete it
        # * swap it with another interval
        # * replace it with another interval
        # * replace it with a new random string
        #
        # for each splice point n1:
        # * with some ~low probability, insert a new random string at n1
        #
        # the latter is not covered by the interval operators as the interval
        # operators have no way to introduce new characters without otherwise
        # modifying the string (but they do for removing characters).
        # ~equivalently this could be an operation on the start point of
        # each interval.
        num_splice = _size(
            min_size=min(2, len(value)),
            max_size=5,
            average_size=min(2.5, len(value)),
            random=random,
        )
        splice_points = sorted(
            {random.randint(0, len(value)) for _ in range(num_splice)}
        )
        splice_intervals = list(zip(splice_points, splice_points[1:]))
        # (n1, n2): new_string. use a dict to allow overwriting operators
        # for the same splice interval lest size changes get the better of us.
        replacements = {}
        for n1, n2 in splice_intervals:
            r = random.randint(0, 3)
            if r == 0:
                # case: delete this interval
                replacements[(n1, n2)] = []
            elif r == 1 and len(splice_intervals) > 1:
                # case: swap with another interval
                (a1, a2) = random.choice(splice_intervals)
                replacements[(n1, n2)] = value[a1:a2]
                replacements[(a1, a2)] = value[n1:n2]
            elif r == 2 and len(splice_intervals) > 1:
                # case: replace with another interval
                (a1, a2) = random.choice(splice_intervals)
                replacements[(n1, n2)] = value[a1:a2]
            elif r == 3:
                # case: replace with a new random string of ~similar size
                replacements[(n1, n2)] = draw_value(
                    min_size=0, average_size=n2 - n1, max_size=(n2 - n1) * 2
                )

        replacements = [(n1, n2, value) for (n1, n2), value in replacements.items()]
        forced = replace_all(value, replacements)
        for n in splice_points:
            if random.randint(0, 10) == 0:
                # case: insert a new random value at point n
                # TODO I think this misses inserting at the very end, see len(forced) + 1
                # case in `while len(forced) < min_size`
                forced = (
                    forced[:n]
                    + draw_value(min_size=0, average_size=2, max_size=6)
                    + forced[n:]
                )

        # if none of this mutation has had any effect, add or delete something
        # random for certain. this can happen for rather small strings where we don't
        # have many/any splice_intervals.
        if forced == value:
            n = random.randint(0, len(value))
            if random.randint(0, 1) == 0 and len(value) > 0:
                # remove everything up to n, or after n
                forced = forced[n:] if random.randint(0, 1) == 0 else forced[:n]
            else:
                # add a string at n
                forced = (
                    forced[:n]
                    + draw_value(min_size=0, average_size=2, max_size=4)
                    + forced[n:]
                )

        while len(forced) < min_size:
            add_idx = random.choice(list(range(len(forced) + 1)))
            forced = forced[:add_idx] + [draw_element()] + forced[add_idx:]
        while len(forced) > max_size:
            # remove random indices to bring us back to max_size
            remove_idx = random.choice(list(range(len(forced))))
            forced = forced[:remove_idx] + forced[remove_idx + 1 :]

    assert min_size <= len(forced) <= max_size
    return forced


# upweight certain ascii chars. copied from libfuzzer with some
# modifications (e.g. drop 0 and 255).
# https://github.com/llvm/llvm-project/blob/0f56ba13bff7ab72bfafcf7c5cf
# 9e5b8bd16d895/compiler-rt/lib/fuzzer/FuzzerMutate.cpp#L66C1-L67C1
#    const char Special[] = "!*'();:@&=+$,/?%#[]012Az-`~.\xff\x00";
upweighted_ascii = [ord(x) for x in "!*'();:@&=+$,/?%#[]012Az-`~."]


def mutate_string(value, *, min_size, max_size, intervals, random):
    # 50% of the time, use pure ascii. 20% of the time use pure random (which may
    # coincidentally generate ascii). the other 30%, mix our ascii upweighting
    # scheme with unicode, where ascii is upweighted 80%.
    #
    # atheris does one more thing we don't, which is a 25% chance of a
    # utf-16 compatible string. I just think it's low-impact and haven't bothered
    # yet.
    # https://github.com/google/atheris/blob/cbf4ad989dcb4d3ef42152990ed89cfceb
    # 50e059/src/native/fuzzed_data_provider.cc#L61
    r = random.random()
    generation_kind = "ascii" if r < 0.5 else "random" if r < 0.7 else "mixed"
    upweighted_choices = [n for n in upweighted_ascii if n < intervals.size]

    def ascii():
        # upweight special ascii chars with 40% probability. libfuzzer does 50%
        # but this just seems too high.
        if upweighted_choices and random.random() < 0.4:
            return random.choice(upweighted_choices)
        # this is actually the first 256 characters, not strictly ascii (first 128).
        # matches hypothesis and I think libfuzzer.
        return random.randint(0, min(intervals.size - 1, 255))

    def random_char():
        return random.randint(0, intervals.size - 1)

    def draw_element():
        if generation_kind == "ascii":
            n = ascii()
        elif generation_kind == "random":
            n = random_char()
        elif generation_kind == "mixed":
            if random.random() < 0.8:
                n = ascii()
            else:
                n = random_char()
        else:
            assert False
        return chr(intervals[n])

    return "".join(
        mutate_collection(
            value,
            min_size=min_size,
            max_size=max_size,
            draw_element=draw_element,
            random=random,
        )
    )


def mutate_bytes(value, *, min_size, max_size, random):
    def draw_element():
        return random.randint(0, 255)

    return bytes(
        mutate_collection(
            value,
            min_size=min_size,
            max_size=max_size,
            draw_element=draw_element,
            random=random,
        )
    )


def _custom_mutator(data, buffer_size, seed):
    return custom_mutator(data, random=Random(seed), blackbox=True)


def custom_mutator(data, *, random, blackbox):
    t_start = time.time()
    statistics["num_calls"] += 1
    # custom_mutator should be called by atheris exactly once per test case.
    global custom_mutator_called, stats_printed
    # this assert actually fired when fuzzing hypothesis_jsonschema? not sure how yet.
    # assert not custom_mutator_called
    custom_mutator_called = True

    stats = {}

    def _fresh():
        stats["mode"] = "fresh"
        # ramp up to buffer_size or pick a weighted length from [0, buffer_size]?
        # returning randbytes(BUFFER_SIZE) probably has performance implications.
        return random.randbytes(BUFFER_SIZE)

    # blackbox exploratory/warmup phase for the first 5k inputs / 5 seconds
    # TODO we probably want an adaptive tradeoff here, "blackbox until n consecutive uninteresting inputs"
    if blackbox and (statistics["num_calls"] < 5000 or statistics["time_mutating"] < 5):
        return _fresh()

    try:
        bounds = data_to_bounds[data]
    except KeyError:
        # we haven't seen this data before. atheris is trying a fresh input.
        return _fresh()
    if track_per_item_stats:
        stats["mode"] = "mutate"

    choices = list(bounds.keys())
    # possibly 0 choices got made, in which case use 0 mutations.
    num_mutations_ = num_mutations(
        min_size=min(1, len(choices)), max_size=len(choices), random=random
    )
    mutations = random.sample(range(len(choices)), num_mutations_)
    if track_per_item_stats:
        stats["num_mutations"] = len(mutations)
        stats["before"] = [_make_serializable(v) for (_, _, v) in bounds.values()]
        stats["mutations"] = []
        after = [_make_serializable(v) for (_, _, v) in bounds.values()]

    def _get_splice_choices(location, ir_type, kwargs):
        splices = []
        for splice_location, (splice_ir_type, _splice_kwargs, value) in bounds.items():
            # don't splice an ir node with itself (same start, end value)
            if location == splice_location:
                continue
            if ir_type != splice_ir_type:
                continue
            if not ir_value_permitted(value, ir_type, kwargs):
                continue
            splices.append(value)
        return splices

    replacements = []
    # TODO with probability 0.2, do only splices (average of 2 maybe?). or sometimes
    # try splice *ranges* - ranges of ir the same ir types
    for i in mutations:
        start, end = choices[i]
        (ir_type, kwargs, value) = bounds[(start, end)]
        # TODO reconsider forced value handling - re-sample from num mutations?
        # or sample by construction to avoid sampling forced nodes?
        if kwargs["forced"] is not None:
            continue

        extra_item_stats = {}
        mutation_type = "normal"
        # 10% chance for a mutation to be a splice (copy) of an existing node of
        # the same type instead
        if random.randint(0, 9) == 0 and (
            splice_choices := _get_splice_choices((start, end), ir_type, kwargs)
        ):
            mutation_type = "splice"
            forced = random.choice(splice_choices)
            if track_per_item_stats:
                extra_item_stats["splice_choices"] = [
                    _make_serializable(v) for v in splice_choices
                ]
        elif ir_type == "integer":
            forced = mutate_integer(
                value,
                min_value=kwargs["min_value"],
                max_value=kwargs["max_value"],
                random=random,
            )
        elif ir_type == "boolean":
            p = kwargs["p"]
            assert 0 < p < 1
            forced = bool(random.randint(0, 1))
        elif ir_type == "bytes":
            forced = mutate_bytes(
                value,
                min_size=kwargs["min_size"],
                max_size=kwargs["max_size"],
                random=random,
            )
        elif ir_type == "string":
            forced = mutate_string(
                value,
                min_size=kwargs["min_size"],
                max_size=kwargs["max_size"],
                intervals=kwargs["intervals"],
                random=random,
            )
        elif ir_type == "float":
            forced = mutate_float(
                value,
                min_value=kwargs["min_value"],
                max_value=kwargs["max_value"],
                allow_nan=kwargs["allow_nan"],
                smallest_nonzero_magnitude=kwargs["smallest_nonzero_magnitude"],
                random=random,
            )
        else:
            raise Exception(f"unhandled case ({ir_type=})")

        # value of the prefix doesn't matter since we're forcing the draw.
        cd = AtherisData(BUFFER_SIZE, prefix=bytes(BUFFER_SIZE), random=random)
        # overwrite the forced val in the kwargs
        kwargs = {**kwargs, "forced": forced}
        try:
            getattr(cd, f"draw_{ir_type}")(**kwargs)
        except StopTest:
            assert cd.status is Status.OVERRUN
            # should hopefully be rare? can happen if we get unlucky with integer
            # probes or if we increase the size of eg a string. replace with randomness
            # in this case I guess?
            replacement = random.randbytes(end - start)
        else:
            replacement = cd.buffer

        if track_per_item_stats:
            after[i] = _make_serializable(forced)
            stats["mutations"].append(
                {
                    "ir_type": ir_type,
                    "before": _make_serializable(value),
                    "after": _make_serializable(forced),
                    "mutation_type": mutation_type,
                    **extra_item_stats,
                }
            )
        replacements.append((start, end, replacement))

    replacements = sorted(replacements, key=lambda v: v[0])
    # some strategies have empty choice sequences with no bounds (st.just(None)).
    offset = max(bounds.keys(), key=lambda v: v[1])[1] if bounds else 0
    for u, v, r in replacements:
        offset += len(r) - (v - u)
    data = bytearray(replace_all(data, replacements))
    # fully randomize each time in case we misalign and dip into the remaining buffer.
    # we don't want to use the same randomized+saved buffer each time.
    # this weirdness is to allow libfuzzer to REDUCE our inputs instead of always
    # being the same BUFFER_SIZE size.
    del data[offset:]
    data += random.randbytes(BUFFER_SIZE // 2)
    if len(data) > BUFFER_SIZE:
        data = data[:BUFFER_SIZE]
    data = bytes(data)
    statistics["time_mutating"] += time.time() - t_start

    if track_per_item_stats:
        stats["after"] = after
        statistics["per_item_stats"].append(stats)

    if statistics["num_calls"] > print_stats_at and not stats_printed:
        import json

        print("-- run statistics --")
        print(json.dumps(statistics))
        stats_printed = True
    return data




class AtherisData(ConjectureData):
    def draw_bits(
        self,
        n: int,
        *,
        forced: Optional[int] = None,
        fake_forced: Optional[bool] = False,
    ) -> int:
        if n == 0:
            return 0
        assert n > 0
        n_bytes = bits_to_bytes(n)
        if self.index + n_bytes > len(self.prefix):
            self.mark_overrun()

        if forced is not None:
            # if we are not asking for a multiple of 8 bits, there are 8 - (n % 8)
            # bits worth of "unused" randomness. Rather than leaving these as 0,
            # we fill these bits randomly. It makes no difference when re-reading
            # the forced value and ensures we maintain full randomness during
            # misalignments.
            m = 0 if n % 8 == 0 else self.random.randint(0, 2 ** (8 - (n % 8)) - 1)
            buf = int_to_bytes(forced | (m << n), n_bytes)
            result = forced
        else:
            buf = self.prefix[self.index : self.index + n_bytes]
            buf = bytearray(buf)
            # If we have a number of bits that is not a multiple of 8 we have to
            # mask off the high bits.
            buf[0] &= BYTE_MASKS[n % 8]
            result = int_from_bytes(buf)

        assert len(buf) == n_bytes
        self.buffer.extend(buf)
        self.index = len(self.buffer)
        assert result.bit_length() <= n
        return result


class AtherisProvider(PrimitiveProvider):
    def draw_value(self, ir_type, kwargs):
        start = self.cd.index
        v = getattr(self.cd, f"draw_{ir_type}")(**kwargs)
        self.bounds[(start, self.cd.index)] = (ir_type, kwargs, v)
        return v

    def draw_boolean(self, **kwargs):
        return self.draw_value("boolean", kwargs)

    def draw_integer(self, **kwargs):
        return self.draw_value("integer", kwargs)

    def draw_float(self, **kwargs):
        return self.draw_value("float", kwargs)

    def draw_string(self, **kwargs):
        return self.draw_value("string", kwargs)

    def draw_bytes(self, **kwargs):
        return self.draw_value("bytes", kwargs)

    @contextmanager
    def per_test_case_context_manager(self):
        # {(start, end): (ir_type, kwargs, value)}
        self.bounds = {}
        self.index = 0
        self.buffer = bytearray()
        # TODO seed? how to get mutator / atheris seed here?
        self.random = Random()
        self.cd = AtherisData(BUFFER_SIZE, prefix=self.prefix, random=self.random)

        yield
        # explicitly don't use try/finally as we don't want to set data_to_bounds
        # in the case of mark_overrun. we may want to change this in the future.
        data_to_bounds[self.prefix] = self.bounds
