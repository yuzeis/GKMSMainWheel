# Copyright (C) 2026 花吹雪また一年P and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""scenario/observe_import.py: a session overwritten with an observed position plays on like the observed one."""
import pytest

from GakumasCLI.scenario.hif import build_hif_loadout_session
from GakumasCLI.scenario.observe_import import SCHEMA, export_observation, import_observation


def _first_turn_plays(session, n):
    for _ in range(n):
        plays = [a for a in sorted(session.legal_actions()) if a[0] == "play"]
        if not plays or session.turn != 1:
            break
        session.step(plays[0])


def _previews(session):
    out = {}
    for action in sorted(session.legal_actions()):
        p = session.preview_action(action)
        out[action] = (p["after_action"]["score"], p["after_action"]["stamina"], p["after_settlement"]["score"])
    return out


def test_import_roundtrip(master, loadouts):
    for kind, lo in loadouts.items():
        a, _ = build_hif_loadout_session(master, lo, seed=3)
        _first_turn_plays(a, 2)
        obs = export_observation(a)
        assert obs["schema"] == SCHEMA
        b, _ = build_hif_loadout_session(master, lo, seed=11)  # another shuffle: another hand and deck order
        import_observation(b, obs)
        assert export_observation(b) == obs, kind
        assert sorted(b.legal_actions()) == sorted(a.legal_actions()), kind
        assert _previews(b) == _previews(a), kind


def test_import_mid_turn_statuses(master, loadouts):
    lo = next(iter(loadouts.values()))
    s, _ = build_hif_loadout_session(master, lo, seed=5)
    obs = {"schema": SCHEMA, "score": 1234, "stamina": 7, "block": 9,
           "statuses": {"parameter_buff": 3, "parameter_buff_multiple": 2, "lesson_buff": 5, "review": 8,
                        "aggressive": 4, "full_power_point": 2},
           "turn_play_count": 1, "playable": 1}
    import_observation(s, obs)
    view = s.observe()
    assert (view["score"], view["stamina"], view["block"]) == (1234, 7, 9)
    assert (view["lesson_buff"], view["review"], view["aggressive"], view["full_power_point"]) == (5, 8, 4, 2)
    assert view["timed_layers"]["parameter_buff_remaining"] == 3
    assert view["turn_play_count"] == 1
    got = export_observation(s)["statuses"]
    assert {k: got[k] for k in obs["statuses"]} == obs["statuses"]
    import_observation(s, {"schema": SCHEMA, "turn": 3, "statuses": {"parameter_buff": 2}})
    assert s.turn == 3 and export_observation(s)["statuses"]["parameter_buff"] == 2
    ends = 0
    while not s.finished:
        s.step(("end",) if ("end",) in s.legal_actions() else sorted(s.legal_actions())[0])
        ends += 1
    assert ends <= int(s.config.limit_turn) - 3 + 1  # the session plays on from turn 3


def test_import_hand_from_deck_and_new_cards(master, loadouts):
    lo = next(iter(loadouts.values()))
    s, _ = build_hif_loadout_session(master, lo, seed=7)
    before = export_observation(s)
    pool = before["deck"] + before["hand"]
    hand = pool[-3:]  # cards now in the deck move to the hand; the rest of the deck keeps its cards
    import_observation(s, {"schema": SCHEMA, "hand": hand})
    after = export_observation(s)
    assert after["hand"] == hand
    key = lambda c: (c["produce_card_id"], c["upgrade"])  # noqa: E731
    assert sorted(map(key, after["hand"] + after["deck"])) == sorted(map(key, pool))
    import_observation(s, {"schema": SCHEMA, "grave": [{"produce_card_id": hand[0]["produce_card_id"], "upgrade": 1}]})
    assert [key(c) for c in export_observation(s)["grave"]] == [(hand[0]["produce_card_id"], 1)]


def test_import_rejects_unknown(master, loadouts):
    lo = next(iter(loadouts.values()))
    s, _ = build_hif_loadout_session(master, lo, seed=1)
    with pytest.raises(ValueError, match="schema"):
        import_observation(s, {"schema": "x"})
    with pytest.raises(ValueError, match="statuses"):
        import_observation(s, {"schema": SCHEMA, "statuses": {"nonsense": 1}})
