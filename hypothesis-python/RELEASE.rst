RELEASE_TYPE: patch

This patch fixes a bug where :func:`~hypothesis.strategies.decimals` with the
``places`` parameter could generate values outside the specified ``min_value``
and ``max_value`` bounds when those bounds had many significant digits
(:issue:`4651`).
