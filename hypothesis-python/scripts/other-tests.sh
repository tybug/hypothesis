#!/bin/bash
set -e -o xtrace

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$HERE/.."

uv pip install .


PYTEST="python -bb -X dev -m pytest -nauto --durations-min=1.0"

# Run some tests without docstrings or assertions, to catch bugs
# like issue #822 in one of the test decorators.  See also #1541.
PYTHONOPTIMIZE=2 $PYTEST \
    -W'ignore:assertions not in test modules or plugins will be ignored because assert statements are not executed by the underlying Python interpreter:pytest.PytestConfigWarning' \
    -W'ignore:Module already imported so cannot be rewritten:pytest.PytestAssertRewriteWarning' \
    tests/cover/test_testdecorators.py

# Run tests for each extra module while the requirements are installed
uv pip install ".[pytz, dateutil, zoneinfo]"
$PYTEST tests/datetime/
uv pip uninstall pytz python-dateutil

uv pip install ".[dpcontracts]"
$PYTEST tests/dpcontracts/
uv pip uninstall dpcontracts

uv pip install attrs
$PYTEST tests/attrs/
uv pip uninstall attrs

# use pinned redis version instead of inheriting from fakeredis
uv pip install "$(grep '^redis==' ../requirements/coverage.txt)"
uv pip install "$(grep 'fakeredis==' ../requirements/coverage.txt)"
uv pip install "$(grep 'typing-extensions==' ../requirements/coverage.txt)"
$PYTEST tests/redis/
uv pip uninstall redis fakeredis

$PYTEST tests/typing_extensions/
if [ "$HYPOTHESIS_PROFILE" != "crosshair" ] && [ "$(python -c 'import sys; print(sys.version_info[:2] > (3, 10))')" = "True" ]; then
  uv pip uninstall typing-extensions
fi

uv pip install "$(grep 'annotated-types==' ../requirements/coverage.txt)"
$PYTEST tests/test_annotated_types.py
uv pip uninstall annotated-types

uv pip install ".[lark]"
uv pip install "$(grep -m 1 -oE 'lark>=([0-9.]+)' ../hypothesis-python/pyproject.toml | tr '>' =)"
$PYTEST -Wignore tests/lark/
uv pip install "$(grep 'lark==' ../requirements/coverage.txt)"
$PYTEST tests/lark/
uv pip uninstall lark

if [ "$(python -c $'import platform, sys; print(sys.version_info.releaselevel == \'final\' and platform.python_implementation() not in ("PyPy", "GraalVM"))')" = "True" ] ; then
  uv pip install ".[codemods,cli]"
  $PYTEST tests/codemods/

  if [ "$(python -c 'import sys; print(sys.version_info[:2] == (3, 10))')" = "True" ] ; then
    # Per NEP-29, this is the last version to support Python 3.10
    uv pip install numpy==2.2.6
  else
    uv pip install "$(grep 'numpy==' ../requirements/coverage.txt)"
  fi

  uv pip install "$(grep -E 'black(==| @)' ../requirements/coverage.txt)"
  $PYTEST tests/patching/
  uv pip uninstall libcst

  # One of the ghostwriter tests uses attrs (though hypothesis[ghostwriter] does not require attrs).
  uv pip install attrs
  $PYTEST tests/ghostwriter/
  uv pip uninstall attrs
  uv pip uninstall black

  if [ "$HYPOTHESIS_PROFILE" != "crosshair" ] ; then
    # Crosshair tracer is not compatible with no-gil
    if [ "$(python -c "import sys; print('free-threading' in sys.version)")" != "True" ] ; then
      # Run twice, interleaved by other tests, to make it more probable to tickle any problems
      # from accidentally caching/retaining crosshair proxy objects
      uv pip install pytest-repeat
      uv pip install -r ../requirements/crosshair.txt
      # requirements/crosshair.txt pins hypothesis. Re-override it with our local changes
      uv pip install .
      $PYTEST --count=2 --repeat-scope=session tests/numpy tests/crosshair
      # ...but running twice takes time, don't overdo it
      $PYTEST tests/array_api
    else
      $PYTEST tests/array_api tests/numpy
    fi
  fi
fi
