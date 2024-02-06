from hypothesis import *
import sys; sys.path.insert(0, "/Users/tybug/Desktop/Liam/coding/hypothesis/hypothesis-python")
from tests.conjecture.common import *
from hypothesis.internal.conjecture.datatree import *
from hypothesis.errors import *
from hypothesis import strategies as st


@given(st.integers())
def f(n):
    assert n < 2
f()
# TODO: test compute_max_children with allow_subnormal
# TODO Branch kwargs needs the float_to_int treatment...consider min_value=0.0 and min_value=-0.0, our kwargs != kwargs check will pass. results in false negatives on flaky?
#      TreeNode(kwargs=[], values=[], ir_types=[], _TreeNode__forced=None, transition=Branch(kwargs={'min_value': -0.0, 'max_value': 4.9e-322, 'allow_nan': True, 'smallest_nonzero_magnitude': 5e-324}, ir_type='float'), is_exhausted=False)
