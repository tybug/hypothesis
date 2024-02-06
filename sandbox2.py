from hypothesis import given, strategies as st
from hypothesis import assume


expression = st.deferred(lambda: st.one_of(
    st.integers(),
    st.tuples(st.just('+'), expression, expression),
    st.tuples(st.just('/'), expression, expression),
))


def div_subterms(e):
    if isinstance(e, int):
        return True
    if e[0] == '/' and e[-1] == 0:
        return False
    return div_subterms(e[1]) and div_subterms(e[2])


def evaluate(e):
    if isinstance(e, int):
        return e
    elif e[0] == '+':
        return evaluate(e[1]) + evaluate(e[2])
    else:
        assert e[0] == '/'
        return evaluate(e[1]) // evaluate(e[2])


@given(expression)
def test(e):
    assume(div_subterms(e))
    try:
        evaluate(e)
    except Exception:
        import traceback; traceback.print_exc()
    evaluate(e)

test()







# The following is a python property-based test using the Hypothesis library:

# from hypothesis import given, strategies as st
# from hypothesis import assume

# expression = st.deferred(lambda: st.one_of(
#     st.integers(),
#     st.tuples(st.just('+'), expression, expression),
#     st.tuples(st.just('/'), expression, expression),
# ))

# def div_subterms(e):
#     if isinstance(e, int):
#         return True
#     if e[0] == '/' and e[-1] == 0:
#         return False
#     return div_subterms(e[1]) and div_subterms(e[2])

# def evaluate(e):
#     if isinstance(e, int):
#         return e
#     elif e[0] == '+':
#         return evaluate(e[1]) + evaluate(e[2])
#     else:
#         assert e[0] == '/'
#     return evaluate(e[1]) // evaluate(e[2])

# @given(expression)
# def test(e):
#     assume(div_subterms(e))
#     evaluate(e)

# There is a bug in this program triggered by the input ('/', -492, ('/', -90, -268)). Here is the stacktrace of the bug:

# Traceback (most recent call last):
# File "/Users/tybug/Desktop/Liam/coding/hypothesis/sandbox2.py", line 34, in test
# evaluate(e)
# File "/Users/tybug/Desktop/Liam/coding/hypothesis/sandbox2.py", line 27, in evaluate
# return evaluate(e[1]) // evaluate(e[2])
# ~~~~~~~~~~~~~~~^^~~~~~~~~~~~~~~~
# ZeroDivisionError: integer division or modulo by zero

# Please suggest a simpler and/or smaller input which would also trigger this bug. Do not give any explanation. Only state the simpler input.




# ==== after ====

# The input {{input}} doesn't trigger the bug. Please suggest another simpler/smaller input which would trigger the bug.
