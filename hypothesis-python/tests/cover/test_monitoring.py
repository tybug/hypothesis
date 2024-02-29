# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from contextlib import contextmanager

import pytest

from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning
from hypothesis.internal.scrutineer import MONITORING_TOOL_ID
import sys

@contextmanager
def using_tool_id(tool_id, tool_name):
    try:
        sys.monitoring.use_tool_id(tool_id, tool_name)
        yield
    finally:
        sys.monitoring.free_tool_id(tool_id)


def test_monitoring_warns_on_registered_tool_id():

    # scrutineer can't run if something has already registered its tool id.
    with using_tool_id(MONITORING_TOOL_ID, "rogue"):
        with pytest.warns(HypothesisWarning, match=r"is already taken by tool rogue"):

            @given(st.integers())
            def f(n):
                assert True

            f()
