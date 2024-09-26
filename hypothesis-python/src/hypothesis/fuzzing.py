# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import inspect
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable, Dict, List, Mapping, TypedDict
import abc
from types import SimpleNamespace

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from hypothesis.database import ir_from_bytes, ir_to_bytes
from hypothesis.errors import StopTest
from hypothesis.internal import floats as flt
from hypothesis.internal.cache import LRUCache
from hypothesis.internal.conjecture.data import (
    COLLECTION_DEFAULT_MAX_SIZE,
    NASTY_FLOATS,
    IRKWargsType,
    IRType,
    IRTypeName,
    PrimitiveProvider,
    ir_to_buffer,
    ir_value_equal,
    ir_value_permitted,
)
from hypothesis.internal.conjecture.engine import BUFFER_SIZE
from hypothesis.internal.conjecture.junkdrawer import clamp
from hypothesis.internal.floats import next_down, next_up, sign_aware_lte
from hypothesis.internal.compat import int_from_bytes

INT_SIZES = (8, 16, 32, 64, 128)
INT_SIZES_WEIGHTS = (4.0, 8.0, 1.0, 1.0, 0.5)
FLOAT_SIZES = (8, 16, 32, 64, 128, 1024)
FLOAT_SIZES_WEIGHTS = (4.0, 8.0, 1.0, 1.0, 0.5, 0.5)
MAX_SERIALIZED_SIZE = BUFFER_SIZE
# explicitly not thread-local so that the watchdog thread can access it
data_to_draws_unsaved: Mapping[bytes, List["Draw"]] = LRUCache(
    15_000, threadlocal=False
)
# stores bounds for interesting / actual corpus data. unbounded
data_to_draws: Dict[bytes, List["Draw"]] = {}


class Statistics(TypedDict):
    per_item_stats: list[Any]
    num_calls: int
    time_mutating: float


statistics: Statistics = {
    "per_item_stats": [],
    "num_calls": 0,
    "time_mutating": 0,
}
track_per_item_stats = False
print_stats_at = 25_000
global_fuzzing_use_ir = True


def _geometric(*, min, average, max, random):
    assert min >= 0, min
    average = clamp(min, average, max)
    if average == 1:
        return 1
    if average < 1:
        return random.randint(0, 1)

    range_size = max - min + 1
    p = 1 / average

    u = random.random()
    x = math.log(1 - u * (1 - (1 - p) ** range_size)) / math.log(1 - p)
    # This is a systematic underestimation of the average in my testing,
    # especially if the bounds are close together.
    #
    # But, it *does* avoid the wrapped/folded distribution of hypothesis'
    # _calculate_p_continue, which folds the prob density of > max_size into
    # max_size itself. I think avoiding this is better for fuzzing.
    return min + math.floor(x)


def random_float_between(min_value, max_value, smallest_nonzero_magnitude, *, random):
    assert not math.isinf(min_value)
    assert not math.isinf(max_value)
    assert sign_aware_lte(min_value, max_value)

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
                if random.random() < 0.5:
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
        # TODO this would be faster with random.getrandbits...
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
        if bits > 18 and random.random() < 0.5:
            bits = min(bits, random.choices(INT_SIZES, INT_SIZES_WEIGHTS, k=1)[0])
            # TODO this would be faster with getrandbits
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
    assert min_value is None or min_value <= forced, (min_value, forced)
    assert max_value is None or forced <= max_value, (max_value, forced)
    return forced


def mutate_integer(value, *, min_value, max_value, random):
    # with some probability, draw in a small area around the previous value.
    if (
        not (
            min_value is not None
            and max_value is not None
            and (max_value - min_value) < 300
        )
        and random.random() < 0.1
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
        and random.random() < 0.1
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
    if random.random() < 0.05 and nasty_floats:
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
        if bits > 18 and random.random() < 7 / 8:
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
        random.random() < 0.05
        and not math.isnan(forced)
        and not math.isinf(forced)
        and permitted(truncated := float(math.floor(forced)))
    ):
        forced = truncated

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

    x = "".join(
        CollectionMutator(
            value=list(value),
            min_size=min_size,
            max_size=max_size,
            draw_element=draw_element,
            random=random,
        ).mutate()
    )
    # print(f"string mutator. {value} |->| {x}")
    return x


def mutate_boolean(value, *, p):
    # dont mutate to an invalid value
    if p == 0:
        return False
    if p == 1:
        return True

    assert 0 < p < 1
    assert value in {True, False}
    return not value


def mutate_bytes(value, *, min_size, max_size, random):
    def draw_element():
        return random.randint(0, 255)

    return bytes(
        CollectionMutator(
            value=list(value),
            min_size=min_size,
            max_size=max_size,
            draw_element=draw_element,
            random=random,
        ).mutate()
    )


def _custom_mutator(data, buffer_size, seed):
    # seeding a random instance is actually not cheap. this random only controls
    # *mutations*, which we don't care about being deterministic (we want *replay*s
    # of buffers to be deterministic). so dont seed.
    return custom_mutator(data, random=Random(), blackbox=True)


def _get_draws_from_cache(buffer: bytes) -> List["Draw"] | None:
    try:
        return data_to_draws[buffer]
    except KeyError:
        try:
            return data_to_draws_unsaved[buffer]
        except KeyError:
            return None


def mutation(*, p: float, repeatable: bool = False) -> Callable:
    def accept(f):
        f._hypothesis_mutation = SimpleNamespace(p=p, repeatable=repeatable)
        return f

    return accept


@dataclass(slots=True)
class Mutation:
    p: int
    func: Callable
    repeatable: bool


@dataclass(slots=True)
class Draw:
    ir_type: IRTypeName
    kwargs: IRKWargsType
    value: IRType
    forced: IRType | None

    def copy(self, *, with_value: IRType = None) -> "Draw":
        if with_value is not None and self.forced is not None:
            assert ir_value_equal(self.ir_type, with_value, self.forced)

        return Draw(
            ir_type=self.ir_type,
            kwargs=self.kwargs,
            value=self.value if with_value is None else with_value,
            forced=self.forced,
        )


class MutationMessage(TypedDict):
    name: str
    p: int
    cost: int | object
    info: dict | None


class Mutator:
    DISABLED = object()
    # class-level variable set by subclasses
    mutations: List[Mutation] = []

    def __init__(self, *, total_cost: int, random: Random):
        self.total_cost = total_cost
        self.random = random
        # shallow copy so we can delete mutations from the list when they get
        # disabled
        self.mutations = self.mutations.copy()
        self.mutation_messages: list[MutationMessage] = []

    def mutate(self):
        # TODO with some low prob, disable some number of random mutations? (swarm testing)
        total_cost = 0
        while total_cost <= (self.total_cost - 1) and self.mutations:
            budget = int(self.total_cost - total_cost)
            assert budget >= 1
            mutation = self.choose_mutation()

            kwargs = {}
            if mutation.repeatable:
                # if this mutation is repeatable, repeat it (potentially many times)
                # with some low p.
                count = 1
                if self.random.random() < 0.05:
                    count += _geometric(min=1, average=10, max=50, random=self.random)
                kwargs["count"] = count

            cost = mutation.func(self, budget=budget, **kwargs)
            self.mutation_messages.append(
                {
                    "name": mutation.func.__name__,
                    "p": mutation.p,
                    "cost": cost,
                    "kwargs": kwargs,
                    "info": {},
                }
            )
            if cost is self.DISABLED:
                self.mutations.remove(mutation)
                continue

            total_cost += cost

        return self.finish()

    def choose_mutation(self) -> Mutation:
        # keep in mind that mutations can be disabled, in which case we want to
        # remove their probability mass. this is why we reconstruct the p list
        # each time.
        return self.random.choices(
            self.mutations, weights=[m.p for m in self.mutations]
        )[0]

    @abc.abstractmethod
    def finish(self):
        pass


class NodeMutator(Mutator):
    mutations: list[Mutation] = []

    def __init__(self, *, total_cost: int, random: Random, draws: list[Draw]):
        super().__init__(total_cost=total_cost, random=random)
        # avoid mutating our saved draws list
        self.draws = draws.copy()
        self.malleable_indices = []

        # we don't want malleable indices to be a dynamic property for speed,
        # but we do need to update it whenever we change the size of the underlying
        # draws list, or move the location of forced nodes.
        self._update_malleable_indices()

    def _update_malleable_indices(self):
        self.malleable_indices = [
            i for i, draw in enumerate(self.draws) if draw.forced is None
        ]

    @mutation(p=0.2)
    def copy_nodes(self, budget: int) -> int | object:
        assert budget >= 1
        if len(self.draws) <= 1:
            return self.DISABLED

        size = _geometric(
            min=1,
            average=4,
            # we don't allow overlapping copies, so the largest we can be is len // 2.
            max=min(20, len(self.draws) // 2, budget),
            random=self.random,
        )
        start = random.randint(0, len(self.draws) - size)

        def _try_find_target_start(size: int) -> int | None:
            target_start = random.randint(0, len(self.draws) - size)
            # dont copy a sequence with itself
            if target_start == start:
                return None

            # don't allow overlapping copies for now. I'm actually not sure if there
            # is a case where an overlapping copy is better than doing a non-overlapping
            # copy instead. If there is, we may want to reconsider this with some p.
            #
            # The only case I can think of is one where a non-overlapping copy is not
            # possible, and so an overlapping copy at least lets you get some benefit.
            # I suppose this is possible? But I think I'd rather bail early in those
            # cases (because we chose a too-large size) and save the mutation budget
            # for elsewhere, including potentially a future copy_nodes with a lower size.
            if abs(target_start - start) < size:
                return None

            for i in range(size):
                draw = self.draws[start + i]
                draw_target = self.draws[target_start + i]
                if draw.ir_type != draw_target.ir_type:
                    return None
                # ir type of both draws is equal
                ir_type = draw_target.ir_type
                if not ir_value_permitted(draw.value, ir_type, draw_target.kwargs):
                    return None

                # forced nodes must match exactly, meaning the target is forced
                # iff the source is forced, and the forced values are equal.
                v1 = draw.forced
                v2 = draw_target.forced
                if v1 is not None or v2 is not None:
                    if v1 is None and v2 is not None:
                        return None
                    if v1 is not None and v2 is None:
                        return None

                    assert v1 is not None
                    assert v2 is not None
                    if not ir_value_equal(ir_type, v1, v2):
                        return None

            return target_start

        def find_target_starts(
            size: int, *, max_targets: int, attempts: int
        ) -> int | None:
            target_starts = set()
            for _ in range(attempts):
                target_start = _try_find_target_start(size)
                if target_start is not None:
                    target_starts.add(target_start)
                if len(target_starts) == max_targets:
                    break
            assert len(target_starts) <= max_targets
            return list(target_starts)

        # now try to find a target sequence with the right ir types and kwargs
        target_starts = find_target_starts(size, max_targets=1, attempts=4)
        # couldn't find a copy target. we haven't changed anything, so the cost is 0.
        if not target_starts:
            return 0

        target_start = target_starts[0]
        for i in range(size):
            value = self.draws[start + i].value
            self.draws[target_start + i] = self.draws[target_start + i].copy(
                with_value=value
            )

        # we changed `size` nodes, which is therefore our cost.
        return size

    @mutation(p=0.2, repeatable=True)
    def repeat_nodes(self, budget: int, count: int) -> int | object:
        # look for a sequence of nodes and then insert `count` copies of it after
        if len(self.draws) <= 1:
            return self.DISABLED

        # size 2 here is an intentional and weird choice. repeating nodes is most
        # (and perhaps exclusively) likely to help with duplicating list elements,
        # which requires at least two nodes: the boolean True and the node after
        # (but an element may also have more than one corresponding node depending
        # on complexity).
        size = _geometric(
            min=2,
            average=4,
            max=min(20, len(self.draws), budget),
            random=self.random,
        )

        if self.random.random() < 0.7 and (
            bool_draws_i := [
                i
                for i, d in enumerate(self.draws)
                if d.ir_type == "boolean"
                and d.value is True
                and i <= len(self.draws) - size
            ]
        ):
            start = self.random.choice(bool_draws_i)
        else:
            start = random.randint(0, len(self.draws) - size)

        copies = []
        for _ in range(count):
            copies += [self.draws[j].copy() for j in range(start, start + size)]
        self.draws = self.draws[: start + size] + copies + self.draws[start + size :]
        self._update_malleable_indices()
        return size * count

    @mutation(p=1, repeatable=True)
    def mutate_node(self, budget: int, count: int) -> int | object:
        assert budget >= 1
        if not self.malleable_indices:
            return self.DISABLED

        def mutate_node(i):
            draw = self.draws[i]
            ir_type = draw.ir_type
            kwargs: Any = draw.kwargs
            if ir_type == "integer":
                value = mutate_integer(
                    draw.value,
                    min_value=kwargs["min_value"],
                    max_value=kwargs["max_value"],
                    random=random,
                )
            elif ir_type == "boolean":
                value = mutate_boolean(draw.value, p=kwargs["p"])
            elif ir_type == "bytes":
                value = mutate_bytes(
                    draw.value,
                    min_size=kwargs["min_size"],
                    max_size=kwargs["max_size"],
                    random=random,
                )
            elif ir_type == "string":
                value = mutate_string(
                    draw.value,
                    min_size=kwargs["min_size"],
                    max_size=kwargs["max_size"],
                    intervals=kwargs["intervals"],
                    random=random,
                )
            elif ir_type == "float":
                value = mutate_float(
                    draw.value,
                    min_value=kwargs["min_value"],
                    max_value=kwargs["max_value"],
                    allow_nan=kwargs["allow_nan"],
                    smallest_nonzero_magnitude=kwargs["smallest_nonzero_magnitude"],
                    random=random,
                )
            else:
                raise Exception(f"unhandled case ({ir_type=})")

            # print(f"{self.draws[i].value} |->| {value}")
            self.draws[i] = self.draws[i].copy(with_value=value)

        def mutate_nodes(indices):
            for i in indices:
                mutate_node(i)

        i = self.random.choice(self.malleable_indices)
        if count == 1:
            mutate_node(i)
            return 1

        # TODO we'd like to expose these as "swarm options" eventually, and leave
        # it to our master algorithm to choose which one to enable or repeat. This
        # requires a relatively rich language for expressing e.g. incompatible
        # swarm options and sub-options for individual mutations, as well as relative
        # probabilities, so I'm not taking it on right now.

        # with equal probability, choose to either mutate nodes of the same chosen
        # type a la swarm testing, or mutate sequential indices.
        # repeating mutate_node arbitrarily is probably not too helpful because it
        # degrades to blackbox.
        if self.random.random() < 0.5:
            # mutate only nodes of the same type a la swarm testing. repeating mutate_node
            # arbitrarily is probably not too helpful because it degrades to blackbox.
            chosen_ir_type = self.draws[i].ir_type
            possible_indices = [
                i
                for i in self.malleable_indices
                if self.draws[i].ir_type == chosen_ir_type
            ]
            count = min(count, len(possible_indices))
            mutate_nodes(self.random.sample(possible_indices, k=count))
            return count
        else:
            # [i, end)
            end = min(i + count, len(self.malleable_indices))
            # avoid mutating forced nodes that may have been between sequential
            # malleable nodes
            mutate_nodes(j for j in range(i, end) if self.draws[j].forced is None)
            return count

    @mutation(p=0.05)
    def delete_nodes(self, budget: int) -> int | object:
        assert budget >= 1
        # dont delete if there's only a single node, because there's no point
        # in having an empty ir tree
        if len(self.draws) <= 1:
            return self.DISABLED

        # upweight the chance that we remove [True, node] as a pair, which corresponds
        # to deleting an element in a list. is this particularly principled? no.
        # do I expect it to work well? yes.
        # (TODO chance for more than one of these? pick a size then while True?)
        size = 2
        if (
            self.random.random() < 0.7
            and (
                bool_draws_i := [
                    i
                    for i, d in enumerate(self.draws)
                    if d.ir_type == "boolean"
                    and d.value is True
                    and i <= len(self.draws) - size
                ]
            )
            and budget >= size
        ):
            bool_draw_i = self.random.choice(bool_draws_i)
            del self.draws[bool_draw_i : bool_draw_i + size]
            self._update_malleable_indices()
            return size

        size = _geometric(
            min=1,
            average=2,
            max=min(5, budget),
            random=self.random,
        )
        n = 0
        while n < size and len(self.draws) > 1:
            del_i = self.random.choice(range(len(self.draws)))
            del self.draws[del_i]
            n += 1

        # we changed the number of draw nodes, so we need to update the malleable
        # indices.
        self._update_malleable_indices()
        return n

    def finish(self):
        return self.draws


class CollectionMutator(Mutator):
    mutations: list[Mutation] = []

    def __init__(
        self,
        *,
        random: Random,
        value: Any,
        min_size: int,
        max_size: int | None,
        draw_element: Callable,
    ):
        total_cost = _geometric(
            min=1,
            average=2.5,
            max=7,
            random=random,
        )
        super().__init__(total_cost=total_cost, random=random)

        if max_size is None:
            max_size = COLLECTION_DEFAULT_MAX_SIZE

        self.value = value
        self.min_size = min_size
        self.max_size = max_size
        self.draw_element = draw_element

    def random_value(self, *, min_size, max_size, average_size):
        size = _geometric(
            min=min_size,
            average=average_size,
            max=max_size,
            random=random,
        )
        return [self.draw_element() for _ in range(size)]

    def _size(self, budget):
        return self.random.randint(1, min(len(self.value), budget))

    @mutation(p=1)
    def delete_interval(self, budget):
        # delete n1:n2
        if len(self.value) == 0:
            return self.DISABLED

        # first pick a size, then pick a valid starting location
        size = self._size(budget)
        n = self.random.randint(0, len(self.value) - size)
        del self.value[n : n + size]
        return size

    @mutation(p=0.5)
    def swap_interval(self, budget):
        # swap n1:n2 with n3:n4
        if len(self.value) == 0:
            return self.DISABLED
        if len(self.value) == 1:
            return 0

        size1 = self._size(budget)
        size2 = self._size(budget)
        n1 = self.random.randint(0, len(self.value) - size1)
        n2 = self.random.randint(0, len(self.value) - size2)
        if n1 == n2:
            # don't swap an interval to itself
            return 0

        self.value[n1 : n1 + size1], self.value[n2 : n2 + size2] = (
            self.value[n2 : n2 + size2],
            self.value[n1 : n1 + size1],
        )
        # print(f"swap interval {n1=} {n2=} {size1=} {size2=}")
        return max(size1, size2)

    @mutation(p=0.5)
    def copy_interval(self, budget):
        # copy n1:n2 to n3:n4
        if len(self.value) == 0:
            return self.DISABLED
        if len(self.value) == 1:
            return 0

        size = self._size(budget)
        # start of the first interval
        n1 = self.random.randint(0, len(self.value) - size)
        # start of the second interval. it's okay if these overlap
        n2 = self.random.randint(0, len(self.value) - size)
        if n1 == n2:
            # don't copy an interval to itself
            return 0

        self.value[n2 : n2 + size] = self.value[n1 : n1 + size]
        # print(f"copy interval {n1=} {n2=} {size=}")
        return size

    @mutation(p=1)
    def replace_interval(self, budget):
        # replace n1:n2 with random values
        if len(self.value) == 0:
            return self.DISABLED

        size = self._size(budget)
        n = self.random.randint(0, len(self.value) - size)
        self.value[n : n + size] = [self.draw_element() for _ in range(size)]
        return size

    @mutation(p=1)
    def insert_value(self, budget):
        # insert a new random value at n
        assert budget >= 1
        n = self.random.randint(0, len(self.value))
        max_size = min(7, budget)
        v = self.random_value(
            min_size=1,
            average_size=min(2, max_size),
            max_size=max_size,
        )
        self.value = self.value[:n] + v + self.value[n:]
        return len(v)

    def finish(self):
        value = self.value
        # add or remove random values to bring us back within the size bound
        while len(value) < self.min_size:
            add_idx = random.choice(list(range(len(value) + 1)))
            value = value[:add_idx] + [self.draw_element()] + value[add_idx:]
        while len(value) > self.max_size:
            remove_idx = random.choice(list(range(len(value))))
            value = value[:remove_idx] + value[remove_idx + 1 :]

        assert self.min_size <= len(value) <= self.max_size
        return value


# inspection is expensive, run only once outside instead of on Mutator init.
# also precalculate cumulative probs for mutation choices
for MutatorClass in [NodeMutator, CollectionMutator]:
    for _name, method in inspect.getmembers(MutatorClass, predicate=inspect.isfunction):
        if not hasattr(method, "_hypothesis_mutation"):
            continue
        data = method._hypothesis_mutation
        MutatorClass.mutations.append(
            Mutation(func=method, p=data.p, repeatable=data.repeatable)
        )


SIZE_UNCAPPED = b"\x00"


def _size_cap(n):
    # - 10 inputs per byte for the first 20 inputs
    # - 5 inputs per byte for the next 50 inputs
    # - 1 input per byte for the next 50 inputs
    # - 5 bytes per input for the next 50 inputs
    # - 20 bytes per input for the next 50 inputs
    size = 1
    if n >= 20:
        size += (n - 20) // 5
    if n >= 70:
        size += n - 70
    if n >= 130:
        size += (n - 130) * 5
    if n >= 180:
        size += (n - 130) * 20
    return size


def _fresh_input(*, size_cap: int | None = None):
    # size_cap must fit inside a single byte, and 0 is our INTERPRET_IR marker
    # so it can't be that
    if size_cap is not None:
        assert 1 <= size_cap <= 255

    # this is only used as the random seed unless it coincidentally is a valid
    # ir bytestream. so we don't need all BUFFER_SIZE randomness here.
    #
    # TODO is this harming things by always being a REDUCE input if applicable?
    # since 10 bytes is so small it will ~always be a reduction
    v = random.randbytes(10)
    prefix = SIZE_UNCAPPED if size_cap is None else size_cap.to_bytes(1, "big")
    return prefix + v


def custom_mutator(data: bytes, *, random: Random, blackbox: bool) -> bytes:
    # print("HYPOTHESIS MUTATING FROM", data)
    t_start = time.time()
    statistics["num_calls"] += 1

    if statistics["num_calls"] % 1000 == 0:
        print(statistics["time_mutating"])

    stats = {}

    # blackbox exploratory/warmup phase for the first set of inputs
    if blackbox and statistics["num_calls"] < 254:
        stats["mode"] = "fresh"
        return _fresh_input(size_cap=statistics["num_calls"])

    # sometimes the entropic scheduler will mutate an input we just generated,
    # but didn't deem interesting. if we still have that in the front cache,
    # great - use it.
    draws = _get_draws_from_cache(data)
    if draws is None:
        # if it's not saved in the main cache and has also expired from the front
        # cache, degrade to blackbox.
        stats["mode"] = "fresh"
        return _fresh_input()

    if track_per_item_stats:
        stats["mode"] = "mutate"

    malleable_draws = [i for i, draw in enumerate(draws) if draw.forced is None]
    # manually specify distributions for small mutations, where our geometric
    # distribution estimation can be badly off skew. We can also more faithfully
    # specify the desired distribution when working with small integers where it
    # matters most.
    #
    # for instance, with 2 nodes, geom probably says something like mutate both
    # with p=0.25. but we'd really like it to be lower than that.
    max_cost = len(malleable_draws)
    average_cost = {
        0: 0,
        1: 1,
        2: 1.6,
        3: 1.9,
        4: 2.2,
        5: 2.4,
        6: 2.6,
        7: 2.7,
        8: 2.8,
        9: 3,
        10: 3.2,
        11: 3.4,
        12: 3.6,
    }.get(max_cost)
    if average_cost is None:
        assert max_cost > 12
        # add 0.2 average size per additional draw
        average_cost = 3.6 + (max_cost - 12) * 0.2
        # ...up to a cap of 20% average of the nodes for large draws
        if max_cost > 50:
            average_cost = min(average_cost, max_cost * 0.2)

    total_cost = _geometric(
        min=0 if max_cost == 0 else 1,
        average=average_cost,
        max=max_cost,
        random=random,
    )

    mutated_draws = NodeMutator(
        total_cost=total_cost, draws=draws, random=random
    ).mutate()
    serialized_ir = ir_to_bytes([draw.value for draw in mutated_draws])
    serialized_ir = serialized_ir[:MAX_SERIALIZED_SIZE]
    assert len(serialized_ir) <= MAX_SERIALIZED_SIZE

    statistics["time_mutating"] += time.time() - t_start
    if track_per_item_stats:
        stats["after"] = [_make_serializable(draw.value) for draw in mutated_draws]
        statistics["per_item_stats"].append(stats)
    # print("HYPOTHESIS MUTATED TO", serialized_ir)
    return SIZE_UNCAPPED + serialized_ir


class AtherisProvider(PrimitiveProvider):
    def _draws_prefix(self, buffer):
        def ir_type(typ):
            return {
                str: "string",
                float: "float",
                int: "integer",
                bool: "boolean",
                bytes: "bytes",
            }[typ]

        try:
            values = ir_from_bytes(buffer)
        except Exception:
            return None
        return [(ir_type(type(v)), v) for v in values]

    def random_value(
        self, ir_type: IRTypeName, kwargs: IRKWargsType, *, prefix=None
    ) -> IRType:
        if self.draws_prefix is not None and self.draws_index >= len(self.draws_prefix):
            # we've overrun our draws. we don't want to abort this test case, but
            # we also don't want to generate something much larger than we had
            # before. use a zero prefix to generate the simplest possible value.
            prefix = bytes(BUFFER_SIZE)

        if (
            self.max_serialized_size is not None
            and self.serialized_size >= self.max_serialized_size
        ):
            prefix = bytes(BUFFER_SIZE)

        try:
            (value, buf) = ir_to_buffer(
                ir_type, kwargs, random=self.random, prefix=prefix
            )
        except StopTest:
            # possible for a fresh data to overrun if we get unlucky with e.g.
            # integer probes.
            assert self._cd is not None
            self._cd.mark_overrun()
        return value

    def _aligned(self, requested_ir_type, requested_kwargs):
        (ir_type, value) = self.draws_prefix[self.draws_index]
        return requested_ir_type == ir_type and ir_value_permitted(
            value, requested_ir_type, requested_kwargs
        )

    def _value(self, ir_type: IRTypeName, kwargs: Any) -> IRType:
        if self.draws_prefix is None:
            # this buffer wasn't in our cache, do total random generation
            return self.random_value(ir_type, kwargs)
        if self.draws_index >= len(self.draws_prefix):
            return self.random_value(ir_type, kwargs)

        if not self._aligned(ir_type, kwargs):
            # try realigning: pop draws until we realign or run out. if we realign,
            # use that index. if we don't, return random.
            while self.draws_index < len(self.draws_prefix) - 1:
                self.draws_index += 1
                if self._aligned(ir_type, kwargs):
                    break
            else:
                return self.random_value(ir_type, kwargs)

        (_, value) = self.draws_prefix[self.draws_index]
        return value

    def draw_value(self, ir_type: IRTypeName, kwargs: Any) -> IRType:
        forced = kwargs.pop("forced")
        kwargs.pop("fake_forced")
        value = self._value(ir_type, kwargs) if forced is None else forced
        assert (
            type(value)
            is {
                "string": str,
                "float": float,
                "integer": int,
                "boolean": bool,
                "bytes": bytes,
            }[ir_type]
        )

        self.draws_index += 1
        self.serialized_size += len(ir_to_bytes([value]))
        if self.serialized_size > MAX_SERIALIZED_SIZE:
            assert self._cd is not None
            self._cd.mark_overrun()
        self.draws.append(
            Draw(ir_type=ir_type, kwargs=kwargs, value=value, forced=forced)
        )
        # print(f"HYPOTHESIS ATHERISPROVIDER DRAW_VALUE {value} (from {ir_type=}, {kwargs=})")
        return value

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
        self.max_serialized_size = None
        if len(self.buffer) > 0:
            prefix = self.buffer[0]
            self.max_serialized_size = (
                None if prefix == int_from_bytes(SIZE_UNCAPPED) else _size_cap(prefix)
            )

        ir_buffer = self.buffer[1:]
        self.draws_prefix = self._draws_prefix(ir_buffer)

        # print("HYPOTHESIS ATHERISPROVIDER DATA", self.buffer)
        # print("HYPOTHESIS ATHERISPROVIDER DRAWS_PREFIX", self.draws_prefix)
        self.draws_index: int = 0
        self.serialized_size: int = 0
        self.draws: List[Draw] = []
        # deterministic generation relative to the buffer, for replaying failures.
        self.random = Random(self.buffer)

        # explicitly don't use try/finally as we don't want to set data_to_draws
        # in the case of mark_overrun. we may want to change this in the future.
        yield

        data_to_draws_unsaved[self.buffer] = self.draws


class CorpusHandler(FileSystemEventHandler):
    def on_created(self, event):
        p = Path(event.src_path)
        if p.is_dir or not p.exists():
            return

        data = p.read_bytes()
        # this is potentially violated if libfuzzer waits longer than
        # data_to_draws_unsaved.max_size inputs to try a new input. I don't
        # fully understand the entropic scheduler but this seems extraordinarily
        # unlikely bordering on impossible. But I've been bitten by impossible
        # things before...
        # assert data in data_to_draws_unsaved, (data, data_to_draws_unsaved)
        if data not in data_to_draws_unsaved:
            print(
                "WARNING: saved corpus file not present in our time-limited "
                f"cache.\ncache size: {len(data_to_draws_unsaved)}\ndata: {data}"
                f"\ncache: {data_to_draws_unsaved}"
            )
            return
        data_to_draws[data] = data_to_draws_unsaved[data]


def watch_directory_for_corpus(p: str | Path) -> None:
    event_handler = CorpusHandler()
    observer = Observer()
    observer.schedule(event_handler, str(p), recursive=False)
    observer.start()


# NEXT:
# * track statistics for the "lineage" of mutations. how many seeds are saved, how often are they mutated against?
# * nested call test case: if a == 1 if b == 2 if c == 3 ... to ensure we aren't doing anything stupid with seed lineage or caching

# TODO check if our coverage calculation is really only calculating the current lib
