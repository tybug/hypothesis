# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from hypothesis import Phase, settings
from hypothesis.database import (
    DirectoryBasedExampleDatabase,
    InMemoryExampleDatabase,
    MultiplexedDatabase,
    _hash,
)

from tests.common.utils import skipif_threading, wait_for
from tests.cover.test_database_backend import _database_conforms_to_listener_api


def atomic_write(path: Path, data: bytes) -> None:
    fd, tmpname = tempfile.mkstemp()
    tmppath = Path(tmpname)
    os.write(fd, data)
    os.close(fd)
    tmppath.rename(path)


# Depending on the underlying filesystem notification system, DirectoryBasedExampleDatabase
# might drop events. If we save a value in a new key, and the event for writing
# the value gets delivered before the event for writing
#
# e.g.
# * FAILED hypothesis-python/tests/watchdog/test_database.py::
#   test_database_listener_multiplexed -
#   Exception: timing out after waiting 60s for condition lambda: events ==
#   [("save", (b"a", b"a"))] * 2
# * FAILED hypothesis-python/tests/watchdog/test_database.py::
#   test_still_listens_if_directory_did_not_exist -
#   Exception: timing out after waiting 60s for condition lambda: len(events) == 1


# we need real time here, not monkeypatched for CI
time_sleep = time.sleep


def test_database_listener_directory():
    _database_conforms_to_listener_api(
        lambda path: DirectoryBasedExampleDatabase(path),
        supports_value_delete=False,
        parent_settings=settings(
            # this test is expensive because we wait between every rule for
            # the filesystem observer to fire.
            max_examples=5,
            stateful_step_count=10,
            # expensive runtime makes shrinking take forever
            phases=set(Phase) - {Phase.shrink},
            deadline=None,
        ),
    )


# seen flaky on test-win; we get *three* of the same save events in the first
# assertion, which...is baffling, and possibly a genuine bug (most likely in
# watchdog).
@skipif_threading  # add_listener is not thread safe because watchdog is not
def test_database_listener_multiplexed(tmp_path):
    db = MultiplexedDatabase(
        InMemoryExampleDatabase(), DirectoryBasedExampleDatabase(tmp_path)
    )
    events = []

    def listener(event):
        events.append(event)

    db.add_listener(listener)
    time_sleep(0.1)

    db.save(b"a", b"a")
    wait_for(lambda: events == [("save", (b"a", b"a"))] * 2, timeout=30)

    db.remove_listener(listener)
    time_sleep(0.1)
    db.delete(b"a", b"a")
    db.save(b"a", b"b")
    wait_for(lambda: events == [("save", (b"a", b"a"))] * 2, timeout=30)

    time_sleep(0.1)
    db.add_listener(listener)
    time_sleep(0.1)

    db.delete(b"a", b"b")
    db.save(b"a", b"c")
    # InMemory database fires immediately, while DirectoryBased has to
    # wait for filesystem listeners. Therefore the events can arrive out of
    # order. Test a weaker multiset property, disregarding ordering.
    wait_for(
        lambda: Counter(events[2:])
        == {
            # InMemory
            ("delete", (b"a", b"b")): 1,
            # DirectoryBased
            ("delete", (b"a", None)): 1,
            # both
            ("save", (b"a", b"c")): 2,
        },
        timeout=30,
    )


@skipif_threading  # add_listener is not thread safe because watchdog is not
def test_database_listener_directory_explicit(tmp_path):
    db = DirectoryBasedExampleDatabase(tmp_path)
    events = []

    def listener(event):
        events.append(event)

    time_sleep(0.1)
    db.add_listener(listener)
    time_sleep(0.1)

    db.save(b"k1", b"v1")
    wait_for(lambda: events == [("save", (b"k1", b"v1"))], timeout=30)

    time_sleep(0.1)
    db.remove_listener(listener)
    time_sleep(0.1)

    db.delete(b"k1", b"v1")
    db.save(b"k1", b"v2")
    wait_for(lambda: events == [("save", (b"k1", b"v1"))], timeout=30)

    time_sleep(0.1)
    db.add_listener(listener)
    time_sleep(0.1)

    db.delete(b"k1", b"v2")
    db.save(b"k1", b"v3")
    wait_for(
        lambda: events[1:]
        == [
            ("delete", (b"k1", None)),
            ("save", (b"k1", b"v3")),
        ],
        timeout=30,
    )

    # moving into a nonexistent key
    db.move(b"k1", b"k2", b"v3")
    time_sleep(0.5)
    # moving back into an existing key
    db.move(b"k2", b"k1", b"v3")
    time_sleep(0.5)

    if sys.platform.startswith("darwin"):
        assert events[3:] == [
            ("delete", (b"k1", b"v3")),
            ("save", (b"k2", b"v3")),
            ("delete", (b"k2", b"v3")),
            ("save", (b"k1", b"v3")),
        ], str(events[3:])
    elif sys.platform.startswith("win"):
        # watchdog fires save/delete events instead of move events on windows.
        # This means we don't broadcast the exact deleted value.
        assert events[3:] == [
            ("delete", (b"k1", None)),
            ("save", (b"k2", b"v3")),
            ("delete", (b"k2", None)),
            ("save", (b"k1", b"v3")),
        ], str(events[3:])
    elif sys.platform.startswith("linux"):
        # move #1
        assert ("save", (b"k2", b"v3")) in events
        # sometimes watchdog fires a move event (= save + delete with value),
        # and other times it fires separate save and delete events (= delete with
        # no value). I think this is due to particulars of what happens when
        # a new directory gets created very close to the time when a file is
        # saved to that directory.
        assert any(("delete", (b"k1", val)) in events for val in [b"v3", None])

        # move #2
        assert ("save", (b"k1", b"v3")) in events
        assert any(("delete", (b"k2", val)) in events for val in [b"v3", None])
    else:
        raise NotImplementedError(f"unknown platform {sys.platform}")


@skipif_threading  # add_listener is not thread safe because watchdog is not
def test_database_listener_directory_move(tmp_path):
    db = DirectoryBasedExampleDatabase(tmp_path)
    events = []

    def listener(event):
        events.append(event)

    # make sure both keys exist and that v1 exists in k1 and not k2
    db.save(b"k1", b"v1")
    db.save(b"k2", b"v_unrelated")

    time_sleep(0.1)
    db.add_listener(listener)
    time_sleep(0.1)

    db.move(b"k1", b"k2", b"v1")
    # events might arrive in either order
    wait_for(
        lambda: set(events)
        == {
            ("save", (b"k2", b"v1")),
            # windows doesn't fire move events, so value is None
            ("delete", (b"k1", None if sys.platform.startswith("win") else b"v1")),
        },
        timeout=30,
    )


@skipif_threading  # add_listener is not thread safe because watchdog is not
def test_still_listens_if_directory_did_not_exist(tmp_path):
    # if we start listening on a nonexistent path, we will create that path and
    # still listen for events
    events = []

    def listener(event):
        events.append(event)

    p = tmp_path / "does_not_exist_yet"
    db = DirectoryBasedExampleDatabase(p)
    assert not p.exists()

    db.add_listener(listener)
    assert p.exists()

    assert not events
    db.save(b"k1", b"v1")
    wait_for(lambda: len(events) == 1, timeout=30)


@skipif_threading
def test_deferred_save_event(tmp_path):
    # in this test:
    # consistently test a deferred save by avoiding db.save():
    # * write the value file, triggering a deferred save
    # * write the metakeys file, flushing the deferred save
    db = DirectoryBasedExampleDatabase(tmp_path)
    events = []
    db.add_listener(events.append)

    key = b"k1"
    value = b"v1"
    key_hash = _hash(key)

    # trigger a deferred save
    (db.path / key_hash).mkdir()
    atomic_write(db.path / key_hash / _hash(value), value)
    wait_for(lambda: key_hash in db._pending_events, timeout=10)
    assert events == []
    assert db._pending_events[key_hash] == [("save", value)]

    # flush the deferred save
    (db.path / db._metakeys_hash).mkdir()
    atomic_write(db.path / db._metakeys_hash / key_hash, key)
    wait_for(lambda: events == [("save", (key, value))], timeout=10)
    assert db._pending_events == {}


def test_deferred_delete_event(tmp_path):
    # in this test:
    # * write the value file, triggering a deferred save
    # * delete the value file, triggering a deferred delete
    # * write the metakeys file, flushing both the save and delete
    db = DirectoryBasedExampleDatabase(tmp_path)
    events = []
    db.add_listener(events.append)

    key = b"k1"
    value = b"v1"
    key_hash = _hash(key)

    # trigger a deferred save
    (db.path / key_hash).mkdir()
    value_file = db.path / key_hash / _hash(value)
    atomic_write(value_file, value)
    wait_for(lambda: key_hash in db._pending_events, timeout=10)

    # trigger a deferred delete (metakey still hasn't been written)
    value_file.unlink()
    wait_for(lambda: len(db._pending_events[key_hash]) == 2, timeout=10)
    assert events == []
    assert db._pending_events[key_hash] == [("save", value), ("delete", None)]

    # flush the deferred events
    (db.path / db._metakeys_hash).mkdir()
    atomic_write(db.path / db._metakeys_hash / key_hash, key)
    wait_for(lambda: len(events) == 2, timeout=10)
    assert events == [("save", (key, value)), ("delete", (key, None))]
    assert db._pending_events == {}


def test_deferred_move_event(tmp_path):
    # * write the value file, triggering a deferred save
    # * move the value file, triggering a deferred delete + save
    # * write the metakeys files, flushing the events
    db = DirectoryBasedExampleDatabase(tmp_path)
    events = []
    db.add_listener(events.append)

    src_key = b"k1"
    dest_key = b"k2"
    value = b"v1"
    src_hash = _hash(src_key)
    dest_hash = _hash(dest_key)

    # trigger a deferred save
    (db.path / src_hash).mkdir()
    src_file = db.path / src_hash / _hash(value)
    atomic_write(src_file, value)
    wait_for(lambda: src_hash in db._pending_events, timeout=10)

    # trigger a deferred delete + save (from the move)
    (db.path / dest_hash).mkdir()
    dest_file = db.path / dest_hash / _hash(value)
    src_file.rename(dest_file)
    wait_for(lambda: dest_hash in db._pending_events, timeout=10)

    # flush the deferred events
    (db.path / db._metakeys_hash).mkdir()
    atomic_write(db.path / db._metakeys_hash / src_hash, src_key)
    atomic_write(db.path / db._metakeys_hash / dest_hash, dest_key)

    wait_for(lambda: len(events) == 3, timeout=10)
    assert ("save", (src_key, value)) in events
    # watchdog may fire move or delete+create depending on platform
    assert ("delete", (src_key, value)) in events or (
        "delete",
        (src_key, None),
    ) in events
    assert ("save", (dest_key, value)) in events
    assert db._pending_events == {}
