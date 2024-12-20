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


class Lockstep(Shrinker):
    def setup(self, shrinker1, shrinker2):
        (v1, v2) = self.current
        self.shrinker1 = shrinker1(v1, lambda v: self._consider(v, i=0))
        self.shrinker2 = shrinker2(v2, lambda v: self._consider(v, i=1))
        self.v1 = None

    def run_step(self):
        self.shrinker1.run_step()

    def _consider(self, value, i):
        if i == 0:
            self.v1 = value
            self.shrinker2.run_step()
            return

        assert i == 1
        assert self.v1 is not None

        ret = self.consider((self.v1, value))
        self.v1 = None
        return ret
