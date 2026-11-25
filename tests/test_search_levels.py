# Copyright (C) 2026 matto49
# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import pytest

from GakumasCLI.scenario.search_context import (AUTO_LEVELS_MIN_DEADLINE, AUTO_LEVELS_MIN_TRANSITIONS,
                                                auto_levels)


def test_auto_levels_by_budget():
    assert auto_levels() == 2
    assert auto_levels(transitions=20_000) == 2
    assert auto_levels(transitions=AUTO_LEVELS_MIN_TRANSITIONS - 1) == 2
    assert auto_levels(transitions=AUTO_LEVELS_MIN_TRANSITIONS) == 3
    assert auto_levels(deadline=0.5) == 2
    assert auto_levels(deadline=AUTO_LEVELS_MIN_DEADLINE) == 3
    # every budget given has to clear its threshold: a short deadline can stop a long transition budget early
    assert auto_levels(transitions=1_000_000, deadline=0.5) == 2
    assert auto_levels(transitions=1_000_000, deadline=5) == 3


def test_search_ab_arms_and_sign_test():
    from GakumasCLI.native.search_ab import parse_arm, sign_p
    assert parse_arm("L3:levels=3,iterations=16,persist_policy=false,variant=nrpa") == (
        "L3", {"levels": 3, "iterations": 16, "persist_policy": False, "variant": "nrpa"})
    assert parse_arm("base:") == ("base", {})
    with pytest.raises(ValueError):
        parse_arm("x:depth=3")
    assert sign_p(0, 0) == 1.0
    assert sign_p(5, 5) == 1.0
    assert sign_p(10, 0) == pytest.approx(2 / 1024)


@pytest.fixture(scope="module")
def searcher(master):
    from GakumasCLI import paths
    from GakumasCLI.native.diffrun import Native
    from GakumasCLI.native.native_search import NativeSearch
    nat = Native(str(paths.NATIVE_LIB))
    with open(paths.APP_ROOT / "data" / "strings.json", encoding="utf-8") as f:
        nat.load_master(master, json.load(f))
    return NativeSearch(nat)


def test_native_search_takes_its_depth_from_the_budget(master, loadouts, searcher):
    from GakumasCLI.scenario.hif import build_hif_loadout_session
    lo = next(iter(loadouts.values()))
    runs = {}
    for name, kw in (("small", dict(transitions=20_000)), ("large", dict(transitions=AUTO_LEVELS_MIN_TRANSITIONS)),
                     ("given", dict(transitions=AUTO_LEVELS_MIN_TRANSITIONS, levels=2))):
        session, _meta = build_hif_loadout_session(master, lo, seed=7)
        path, st = searcher.policy_search(session, seed=182, **kw)
        assert st["status"] == 0 and path, name
        runs[name] = st
    assert runs["small"]["levels"] == 2
    assert runs["large"]["levels"] == 3
    assert runs["given"]["levels"] == 2
