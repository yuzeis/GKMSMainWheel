# Copyright (C) 2026 花吹雪また一年P and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An observed exam position written into a native session (``gkms.exam.observe_import.v1``).

A client that only sees the screen (no save, no seed) follows the exam on a twin session opened from the same
loadout, steps it with the actions it executed, and overwrites what it observes with ``import_observation``; the
parts it cannot see (the deck order) it samples into several worlds and searches each.  What it does not observe it
leaves out: every key below is optional and an absent key leaves the session's value alone.

    {"schema": "gkms.exam.observe_import.v1",
     "turn": 3,                                   # the session's turn (turn-start effects are not replayed)
     "score": 1234, "stamina": 20, "block": 5,    # block = 元気
     "playable": 1, "turn_play_count": 0,         # actions left this turn, cards played this turn
     "hand": [{"produce_card_id": "p_card-...", "upgrade": 1}, ...],   # in screen order
     "hold": [...], "grave": [...], "lost": [...],                    # 保留 / 捨て札 / 除外
     "deck": [...],                               # 山札, the next draw first
     "statuses": {"parameter_buff": 3,            # 好調 turns
                  "parameter_buff_multiple": 2,   # 絶好調 turns
                  "lesson_buff": 5,               # 集中
                  "review": 8,                    # 好印象
                  "aggressive": 4,                # やる気
                  "full_power_point": 2,          # 全力値
                  "stance": "Concentration",      # 指針: None / Concentration / Preservation / FullPower / Leisurely / OverPreservation
                  "stance_step": 1},
     "drinks": ["pdrink_..."],
     "turn_parameter_types": [2, 3, 3, 1, ...]}    # 審査日程: 1 Vo / 2 Da / 3 Vi for turns 1..N (replaces the opening's draw)

Cards: each listed zone is rewritten in the listed order.  A listed card takes a card the session already has with the
same id and displayed upgrade (current_upgrade_count) -- first one in the same zone, else one from a zone that is not
listed, else any not taken yet -- so a card keeps its own state (play counts, enchants); a card the session does not
have is made by the core (gk_new_card, base upgrade = the displayed one).  A card of a listed zone that no listed card
took moves to the deck when the deck is not listed, else to the grave when that is not listed, else it leaves the
exam.  A zone not listed keeps its cards in their order, less the ones taken.
"""
from __future__ import annotations

from GakumasCLI.native import layout as L

SCHEMA = "gkms.exam.observe_import.v1"
ZONES = ("hand", "hold", "grave", "lost", "deck")  # the order listed cards take
SCALARS = {"score": "score", "stamina": "stamina", "block": "block", "playable": "playable",
           "turn_play_count": "turn_play_count"}
STATUSES = {"lesson_buff": "lesson_buff", "review": "review", "aggressive": "aggressive",
            "full_power_point": "full_power_point", "stance_step": "idol_status_step_count"}
SPECIAL = ("parameter_buff", "parameter_buff_multiple", "stance")
KEYS = {"schema", "turn", "statuses", "drinks", "turn_parameter_types", *SCALARS, *ZONES}


def _key(card):
    return str(card["produce_card_id"]), int(card.get("upgrade", card.get("current_upgrade_count", 0)) or 0)


def _check(obs):
    if obs.get("schema") != SCHEMA:
        raise ValueError(f"observe_import: schema {obs.get('schema')!r}, want {SCHEMA}")
    unknown = set(obs) - KEYS
    if unknown:
        raise ValueError(f"observe_import: unknown keys {sorted(unknown)}")
    bad = set(obs.get("statuses") or {}) - set(STATUSES) - set(SPECIAL)
    if bad:
        raise ValueError(f"observe_import: unknown statuses {sorted(bad)}")
    tpt = obs.get("turn_parameter_types")
    if tpt is not None and (not tpt or len(tpt) > len(L.GkmsState().d_turn_parameter_types)
                            or any(int(t) not in (1, 2, 3) for t in tpt)):
        raise ValueError(f"observe_import: turn_parameter_types {tpt!r}")
    stance = (obs.get("statuses") or {}).get("stance", None)
    if stance is not None and stance not in L._STATUS_TO_CODE:
        raise ValueError(f"observe_import: stance {stance!r}")


def _piles(view, obs):
    """The listed zones rewritten on the edit view's piles (dicts with ``_slot``)."""
    listed = [z for z in ZONES if obs.get(z) is not None]
    if not listed:
        return
    deck = view.deck
    old = {z: list(getattr(deck, z)) for z in ZONES}
    taken = set()

    def take(want, zones):
        for z in zones:
            for c in old[z]:
                if id(c) not in taken and _key(c) == want:
                    taken.add(id(c))
                    return c
        return None

    unlisted = [z for z in ("deck", "grave", "hold", "lost", "hand") if z not in listed]
    new = {}
    for z in listed:
        want = [_key(c) for c in obs[z]]
        out = [take(k, [z]) for k in want]  # the same zone first, for every card of the zone
        new[z] = out
    for z in listed:
        for i, k in enumerate(_key(c) for c in obs[z]):
            if new[z][i] is None:
                new[z][i] = take(k, unlisted) or take(k, listed) or deck._generated_card(*k)
    for z in unlisted:
        new[z] = [c for c in old[z] if id(c) not in taken]
    left = [c for z in listed for c in old[z] if id(c) not in taken]
    if left:
        home = "deck" if "deck" in unlisted else "grave" if "grave" in unlisted else None
        if home is not None:
            new[home] = new[home] + left
    for z in ZONES:
        setattr(deck, z, new[z])


def _turn_types(g, types):
    g.d_has_turn_parameter_types, g.d_n_turn_parameter_types = 1, len(types)
    for i, t in enumerate(types):
        g.d_turn_parameter_types[i] = int(t)


def _scalars(view, obs):
    st = view.g.st
    if obs.get("turn_parameter_types") is not None:
        _turn_types(view.g, obs["turn_parameter_types"])
    turn = obs.get("turn")
    moved = turn is not None and int(turn) != int(view.g.s_turn)
    if moved:
        view.g.s_turn = st.turn = view.g.d_current_turn = int(turn)
    turn = int(st.turn)
    if (moved or obs.get("turn_parameter_types") is not None) and \
            view.g.d_has_turn_parameter_types and 0 < turn <= view.g.d_n_turn_parameter_types:
        view.g.d_current_parameter_type = st.current_parameter_type = view.g.d_turn_parameter_types[turn - 1]
    for k, name in SCALARS.items():
        if obs.get(k) is not None:
            setattr(st, name, int(obs[k]))
    sts = obs.get("statuses") or {}
    for k, name in STATUSES.items():
        if sts.get(k) is not None:
            setattr(st, name, int(sts[k]))
    if sts.get("parameter_buff") is not None:
        n = max(0, int(sts["parameter_buff"]))
        st.parameter_buff_remaining = n
        st.parameter_buff_until_turn = turn + n - 1 if n else L.NONE_I64
        if not n:
            st.parameter_buff_passing_turn_start = 0
    if sts.get("parameter_buff_multiple") is not None:
        n = max(0, int(sts["parameter_buff_multiple"]))
        st.parameter_buff_multiple_until_turn = turn + n - 1 if n else L.NONE_I64
        if not n:
            st.parameter_buff_multiple_per_turn = 0
    if "stance" in sts:
        st.idol_status_type = L._STATUS_TO_CODE[sts["stance"]]
        if sts["stance"] is None and sts.get("stance_step") is None:
            st.idol_status_step_count = 0


def _apply(view, obs):
    _piles(view, obs)
    _scalars(view, obs)


def import_observation(session, obs):
    """``obs`` (``gkms.exam.observe_import.v1``) written into the native ``session``; returns the session."""
    _check(obs)
    if not hasattr(session, "edit") or not hasattr(session, "_export"):
        raise TypeError("observe_import needs a native session")
    session.edit(_apply, obs)
    if obs.get("drinks") is not None:
        session.drinks = list(obs["drinks"])
    return session


def _card(c):
    return {"produce_card_id": c["produce_card_id"], "upgrade": int(c.get("current_upgrade_count") or 0)}


def export_observation(session):
    """What ``import_observation`` takes, read off ``session`` (every key, the deck in draw order)."""
    g = L.GkmsState.from_buffer_copy(memoryview(session._export()).cast("B"))
    st = g.st
    turn = int(session.turn)
    pbm = int(st.parameter_buff_multiple_until_turn)
    view = session.observe()
    out = {"schema": SCHEMA, "turn": turn, **{k: int(getattr(st, n)) for k, n in SCALARS.items()},
           **{z: [_card(c) for c in session._zone(z)] for z in ZONES},
           "statuses": {**{k: int(getattr(st, n)) for k, n in STATUSES.items()},
                        "parameter_buff": int(st.parameter_buff_remaining),
                        "parameter_buff_multiple": 0 if pbm == L.NONE_I64 else max(0, pbm - turn + 1),
                        "stance": view["stance"]},
           "drinks": list(session.drinks),
           "turn_parameter_types": [int(g.d_turn_parameter_types[i]) for i in range(g.d_n_turn_parameter_types)]
           if g.d_has_turn_parameter_types else None}
    return out


__all__ = ["SCHEMA", "export_observation", "import_observation"]
