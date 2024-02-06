from typing import TYPE_CHECKING, Callable, Optional

from hypothesis.internal.conjecture.data import (
    ConjectureData,
)

if TYPE_CHECKING:
    from hypothesis.internal.conjecture.engine import ConjectureRunner

def primitives_for_buffer()

class Shrinker:
    def __init__(
        self,
        engine: "ConjectureRunner",
        initial: ConjectureData,
        predicate: Optional[Callable[..., bool]],
        *,
        allow_transition: bool,
        explain: bool,
        in_target_phase: bool = False,
    ):
        self.engine = engine
        self.shrink_target = initial
        self.predicate = predicate or (lambda data: True)
        self.allow_transition = allow_transition or (lambda source, destination: True)
        self.should_explain = explain

    def shrink(self):
        print(self.shrink_target.buffer)
        ConjectureData.primitives_for_buffer
