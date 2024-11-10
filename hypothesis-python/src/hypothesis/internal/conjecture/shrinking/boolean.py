# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.internal.conjecture.shrinking.common import Shrinker


class Boolean(Shrinker):
    def check_invariants(self, value):
        # must be True, otherwise would be trivial and not selected.
        assert value is True

    def left_is_better(self, left, right):
        return left is False and right is True

    def run_step(self):
        self.consider(False)
