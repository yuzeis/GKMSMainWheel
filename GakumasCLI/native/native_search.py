# Copyright (C) 2026 花吹雪また一年P
# SPDX-License-Identifier: AGPL-3.0-or-later
"""native GNRPA (native/core_search.h) from Python.

    python -B GakumasCLI/gkms.py native.native_search --engine <repo> --plain <plain> --lib <so/dll>
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import sys
import time
from pathlib import Path

S_MAXPATH, S_MAXCURVE = 160, 512


class PriorTable(C.Structure):
    _fields_ = [("has_prior", C.c_int32), ("end_w", C.c_double * 4), ("card_w", C.POINTER(C.c_double)),
                ("n_drink_keys", C.c_int32), ("drink_sid", C.c_int32 * 16), ("drink_n", C.c_int32 * 16),
                ("drink_w", (C.c_double * 4) * 16),
                ("drink_wt", (C.c_double * 16) * 16)]


class SearchParams(C.Structure):
    _fields_ = [("seed", C.c_int64), ("variant", C.c_int32), ("levels", C.c_int32), ("iterations", C.c_int32),
                ("stabilization", C.c_int32), ("repetition_limit", C.c_int32), ("persist_policy", C.c_int32),
                ("transitions", C.c_int64), ("deadline", C.c_double),
                ("lesson_stamina", C.c_int32),
                ("pick_variants", C.c_int32)]


class SearchResult(C.Structure):
    _fields_ = [("status", C.c_int32), ("unsup_line", C.c_int32), ("unsup_info", C.c_int32),
                ("best_score", C.c_int64), ("n_path", C.c_int32), ("path", C.c_uint16 * S_MAXPATH),
                ("expansions", C.c_int64), ("rollouts", C.c_int64), ("incomplete_rollouts", C.c_int64),
                ("adaptation_choices", C.c_int64), ("incumbent_updates", C.c_int64),
                ("top_level_calls", C.c_int64), ("top_level_adaptations", C.c_int64),
                ("repetition_stops", C.c_int64),
                ("n_curve", C.c_int32), ("curve_score", C.c_int64 * S_MAXCURVE),
                ("curve_transitions", C.c_int64 * S_MAXCURVE), ("curve_seconds", C.c_double * S_MAXCURVE),
                ("n_keys", C.c_int32),
                ("best_major", C.c_int64)]


VARIANT = {"random": 0, "nrpa": 1, "gnrpa": 2}
_FEATURE_EPOCH = [0]

# a timeline's rules (native/core_rules.h)
RT_MAX, RT_MAXKEYS = 64, 48
MAJOR_SCALE = 1 << 64
_RT_MODE = {"forbid": 0, "require": 1, "first": 2, "prefer": 3, "cap": 4, "ms_hard": 5, "ms_soft": 6, "no_skip": 7,
            "ms_end": 8, "ms_goal": 9}
_RT_TARGET = {"drink": 0, "end": 1, "play": 2, "none": 3}
_RT_FIELD = {"score": 0, "stamina": 1, "block": 2, "lesson_buff": 3, "review": 4, "aggressive": 5,
             "full_power_point": 6, "lost_ssr": 7, "debuffs": 8,
             "share_vocal": 9, "share_dance": 10, "share_visual": 11}
_RT_FIELD_RULES = {"lost_ssr": 28, "debuffs": 28}
_RT_TIMELINE_2 = ("ms_end", "share_vocal", "share_dance", "share_visual")
_RT_TIMELINE_3 = ("ms_goal",)


def timeline_version(lib):
    """The loaded core's gk_timeline_version (0 when it has none)."""
    try:
        return int(lib.gk_timeline_version())
    except AttributeError:
        return 0


class RuleOne(C.Structure):
    _fields_ = [("abs_mask", C.c_uint64), ("left_mask", C.c_uint64), ("mode", C.c_int32), ("target", C.c_int32),
                ("n_keys", C.c_int32), ("key", C.c_int32 * RT_MAXKEYS), ("upg", C.c_int32 * RT_MAXKEYS),
                ("n", C.c_int32), ("field", C.c_int32), ("op", C.c_int32), ("value", C.c_int64),
                ("weight", C.c_double), ("penalty", C.c_int64)]


class RuleStateC(C.Structure):
    _fields_ = [("turn", C.c_int64), ("acts", C.c_int64), ("plays", C.c_int64), ("done", C.c_uint64),
                ("failed", C.c_uint64), ("pen", C.c_int64)]


class RuleTable(C.Structure):
    _fields_ = [("n", C.c_int32), ("version", C.c_int32), ("has_prefer", C.c_int32), ("has_milestones", C.c_int32),
                ("r", RuleOne * RT_MAX), ("root", RuleStateC)]


def has_rules(lib):
    """Whether the loaded core takes a timeline's rules."""
    try:
        return lib.gk_timeline_version() >= 1 and lib.gk_rule_table_size() == C.sizeof(RuleTable)
    except AttributeError:
        return False


def rule_table(ruleset, index, exam_rules_version=None, timeline_version=None):
    """The native RuleTable of a scenario.timeline.RuleSet; ``index``: string -> string-table index (KeyError for an id
    the table does not know: the caller falls back to the Python search).
    """
    rules = ruleset.rules
    for r in rules:
        if (r.get("mode") in _RT_TIMELINE_2 or r.get("field") in _RT_TIMELINE_2) and int(timeline_version or 1) < 2:
            raise ValueError("this core takes no 控分 rule (gk_timeline_version 2)")
        if r.get("mode") in _RT_TIMELINE_3 and int(timeline_version or 1) < 3:
            raise ValueError("this core takes no goal rule (gk_timeline_version 3)")
        need = _RT_FIELD_RULES.get(r.get("field") or "")
        if need and int(exam_rules_version or 0) < need:
            raise ValueError(f"this core reads no '{r.get('field')}' milestone (exam rules {need})")
    if len(rules) > RT_MAX:
        raise ValueError("too many rules")
    t = RuleTable()
    t.n, t.version = len(rules), 1
    t.has_prefer, t.has_milestones = int(ruleset.has_prefer), int(ruleset.has_milestones)
    for k, r in enumerate(rules):
        o = t.r[k]
        o.abs_mask, o.left_mask = int(r.get("abs", 0)) & ((1 << 64) - 1), int(r.get("left", 0)) & ((1 << 64) - 1)
        o.mode, o.target = _RT_MODE[r["mode"]], _RT_TARGET[r["target"]]
        keys = r.get("keys") or []
        if len(keys) > RT_MAXKEYS:
            raise ValueError("too many keys in a rule")
        o.n_keys = len(keys)
        for j, (key, upg) in enumerate(keys):
            o.key[j], o.upg[j] = index[key], int(upg)
        o.n = int(r.get("n", 0))
        o.field = _RT_FIELD.get(r.get("field") or "score", 0)
        o.op = 0 if (r.get("op") or ">=") == ">=" else 1
        o.value, o.weight, o.penalty = int(r.get("value", 0)), float(r.get("weight", 0.0)), int(r.get("penalty", 0))
    st = ruleset.root
    t.root.turn, t.root.acts, t.root.plays = int(st.turn), int(st.acts), int(st.plays)
    t.root.done, t.root.failed, t.root.pen = int(st.done), int(st.failed), int(st.pen)
    return t


def decode_action(code):
    kind, idx = code >> 8, code & 0xFF
    return ("drink", idx) if kind == 0 else ("end",) if kind == 1 else ("play", idx)


def decode_actions(code):
    """path code kind 3 (GKMS_NATIVE_DRINK_MACRO) is a drink and the play right after it."""
    if code >> 8 == 3:
        idx = code & 0xFF
        return (("drink", idx >> 4), ("play", idx & 15))
    if code >> 8 >= 8:
        idx = code & 0xFF
        return (("select", [(code >> 8) - 8]), ("play" if idx >> 7 else "drink", idx & 15))
    return (decode_action(code),)


class NativeSearch:
    def __init__(self, nat):
        self.nat = nat
        lib = nat.lib
        lib.gk_policy_search.argtypes = [C.c_void_p, C.POINTER(SearchParams), C.POINTER(PriorTable),
                                         C.POINTER(SearchResult)]
        assert lib.gk_search_result_size() == C.sizeof(SearchResult), "SearchResult layout"
        assert lib.gk_search_params_size() == C.sizeof(SearchParams), "SearchParams layout"
        assert lib.gk_prior_table_size() == C.sizeof(PriorTable), "PriorTable layout"
        self.rules_ok = has_rules(lib)
        if self.rules_ok:
            lib.gk_policy_search_rules.argtypes = [C.c_void_p, C.POINTER(SearchParams), C.POINTER(PriorTable),
                                                   C.POINTER(RuleTable), C.POINTER(SearchResult)]

    # -- prior -----------------------------------------------------------
    def prior_table(self, session, prior, nat=None):
        """Resolved prior weights for every master card row x phase, every
        (drink, held count) of this round x phase and 'end' x phase.
        """
        from GakumasCLI.scenario import policy_search as PS
        if getattr(self, "_feature_master", None) is not session.master:
            PS._FEATURE_CACHE.clear()
            self._feature_master = session.master
            _FEATURE_EPOCH[0] += 1
        t = PriorTable()
        if prior is None:
            t.has_prior = 0
            return t, None
        weights = prior.weights
        default = 2.0 * math.log(0.5)
        for z in ("deck", "hand", "grave", "hold", "lost"):
            for card in getattr(session.deck, z):
                PS._card_feature(session, card)
        nat = self.nat if nat is None else nat
        md = nat.md
        n_rows = md.n_cards
        arr = (C.c_double * (4 * n_rows))()
        strings = nat.strings
        for i in range(n_rows):
            row = md.cards[i]
            cid, upg = strings[row.id], int(row.upg)
            fake = {"produce_card_id": cid, "base_upgrade_count": upg, "current_upgrade_count": upg,
                    "temporary_upgrade_count": 0, "support_upgrade_ids": [], "is_end_turn_lost": False}
            feat = PS._FEATURE_CACHE.get((cid, upg))
            if feat is None:
                try:
                    feat = PS._card_feature(session, fake)
                except Exception:   # noqa: BLE001
                    feat = None
            for phase in range(4):
                w = weights.get((phase, cid, upg))
                if w is None:
                    fbs = [("c2", cid, upg), ("c1", cid)]
                    if feat is not None:
                        fbs += [("cf", feat, phase), ("cf0", feat)]
                    for fb in fbs:
                        w = weights.get(fb)
                        if w is not None:
                            break
                arr[4 * i + phase] = default if w is None else w
        t.card_w = C.cast(arr, C.POINTER(C.c_double))
        drinks = list(dict.fromkeys(session.drinks))
        k = 0
        for did in drinks:
            for n in range(1, len(session.drinks) + 1):
                if k >= 16:
                    raise ValueError("prior table: too many drink keys")
                t.drink_sid[k] = nat.index[did]
                t.drink_n[k] = n
                for phase in range(4):
                    w = weights.get((phase, "drink", did, n))
                    if w is None:
                        for fb in (("d2", did), ("d1", phase)):
                            w = weights.get(fb)
                            if w is not None:
                                break
                    t.drink_w[k][phase] = default if w is None else w
                for left in range(16):
                    w = weights.get(("dt", left, did, n))
                    t.drink_wt[k][left] = math.nan if w is None else w
                k += 1
        t.n_drink_keys = k
        for phase in range(4):
            w = weights.get((phase, "end"))
            t.end_w[phase] = default if w is None else w
        t.has_prior = 1
        return t, arr

    def root_prior_table(self, session, prior, nat):
        """``prior_table`` for a search from a native session's own core, kept while the root stays the same -- a job's
        runs (17 of 500,000 transitions in production) search one root on consecutive seeds, and each rebuilt the same
        table (10-13 ms a run on the bench's 本戦 problems).
        """
        if prior is None:
            return self.prior_table(session, prior, nat)
        hit = getattr(self, "_root_prior", None)
        if (hit is not None and hit[0] is prior and hit[1] is nat.md and hit[2] is session.master
                and getattr(self, "_feature_master", None) is session.master
                and hit[3] == (_FEATURE_EPOCH[0], tuple(session.drinks), session.search_key())):
            return hit[4], hit[5]
        table, keep = self.prior_table(session, prior, nat)
        self._root_prior = (prior, nat.md, session.master,
                            (_FEATURE_EPOCH[0], tuple(session.drinks), session.search_key()), table, keep)
        return table, keep

    # -- search ----------------------------------------------------------
    def policy_search(self, session, *, deadline=None, seed=182, variant="gnrpa", transitions=None, levels=None,
                      iterations=8, prior=None, stabilization=1, repetition_limit=0, persist_policy=True,
                      import_state=True, rules=None, pick_variants=0):
        """``rules``: a scenario.timeline.RuleSet; the core must take rules (``rules_ok``), else RuntimeError -- the
        caller falls back to the Python search.  ``levels`` None: by the budget (scenario.search_context.auto_levels),
        the chosen depth is in the stats as ``levels``.
        """
        if levels is None:
            from GakumasCLI.scenario.search_context import auto_levels
            levels = auto_levels(transitions, deadline)
        if rules is not None and not self.rules_ok:
            raise RuntimeError("this core takes no timeline rules")
        if variant not in VARIANT:
            raise ValueError(variant)
        if int(levels) < 1 or int(iterations) < 1:
            raise ValueError("positive nesting parameters required")
        if int(repetition_limit) < 0 or int(stabilization) < 1:
            raise ValueError("repetition_limit must be >= 0 and stabilization >= 1")
        if deadline is None and transitions is None:
            raise ValueError("stochastic research requires a deadline or transition budget")
        prep_started = time.perf_counter()
        from GakumasCLI.native import portable as P
        own = bool(getattr(session, "native", False) and getattr(session, "shadow", None) is None)
        nat = session.search_tables() if own else self.nat
        try:
            cached = getattr(self, "_cache", None)
            if own:
                table, keep = self.root_prior_table(session, prior, nat)
            elif (cached is not None and cached[0] is session and cached[1] is prior
                    and cached[4] is self.nat.md and self.nat.source_master is session.master):
                table = cached[2]
            else:
                if import_state:
                    self.nat.load_round(session)
                    self.nat.import_bytes(P.encode_session(session))
                table, keep = self.prior_table(session, prior)
                self._cache = (session, prior, table, keep, self.nat.md)
            p = SearchParams(seed=int(seed), variant=VARIANT[variant], levels=int(levels), iterations=int(iterations),
                             stabilization=int(stabilization), repetition_limit=int(repetition_limit),
                             persist_policy=int(bool(persist_policy)),
                             transitions=-1 if transitions is None else int(transitions),
                             deadline=-1.0 if deadline is None else float(deadline),
                             lesson_stamina=int(str(getattr(getattr(session, "config", None), "segment_type", ""))
                                                .startswith("Lesson")),
                             pick_variants=max(0, int(pick_variants or 0)))
            r = SearchResult()
            t0 = time.perf_counter()
            if rules is not None:
                rt = rule_table(rules, nat.index, getattr(self.nat, "exam_rules_version", 0),
                                timeline_version(self.nat.lib))
                rc = self.nat.lib.gk_policy_search_rules(nat.core, C.byref(p), C.byref(table), C.byref(rt),
                                                         C.byref(r))
            else:
                rc = self.nat.lib.gk_policy_search(nat.core, C.byref(p), C.byref(table), C.byref(r))
        finally:
            if own:
                nat.free()
        elapsed = time.perf_counter() - t0
        path = tuple(a for i in range(r.n_path) for a in decode_actions(r.path[i]))
        stats = {k: int(getattr(r, k)) for k in ("expansions", "rollouts", "incomplete_rollouts",
                                                "adaptation_choices", "incumbent_updates", "top_level_calls",
                                                "top_level_adaptations", "repetition_stops", "n_keys")}
        best = int(r.best_score) - int(r.best_major) * MAJOR_SCALE
        stats.update(status=int(rc), best_score=best, elapsed_seconds=elapsed, levels=int(levels),
                     prep_seconds=round(t0 - prep_started, 4),
                     unsup_line=int(r.unsup_line), unsup_info=int(r.unsup_info),
                     curve=[dict(score=int(r.curve_score[i]), transitions=int(r.curve_transitions[i]),
                                 seconds=float(r.curve_seconds[i])) for i in range(r.n_curve)])
        return path, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--strings", required=True)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--transitions", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[182])
    ap.add_argument("--stabilization", type=int, default=3)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    engine = str(Path(a.engine).resolve())
    plain, lib = str(Path(a.plain).resolve()), str(Path(a.lib).resolve())
    prior_path = str(Path(a.prior).resolve()) if a.prior else None
    strings = json.load(open(a.strings, encoding="utf-8"))
    os.chdir(engine)
    from GakumasCLI.scenario.hif_replay_import import load_plain_rounds as load_rounds
    from GakumasCLI.native.diffrun import Native
    from GakumasCLI.scenario.exam_session import Master
    from GakumasCLI.scenario import policy_search as PS
    from GakumasCLI.scenario import hif_route_search as HRS
    master = Master.load(Path("ro/master/tables"))
    prior = PS.load_prior(prior_path) if prior_path else None
    nat = Native(lib)
    nat.load_master(master, strings)
    ns = NativeSearch(nat)
    rows = load_rounds(plain)
    keys = sorted(rows)[a.skip:a.skip + a.limit]
    out = []
    for key in keys:
        row = rows[key]
        session, _route = HRS.build(master, row, "fair")
        for seed in a.seeds:
            PS._FEATURE_CACHE.clear()
            ns._cache = None
            npath, nstats = ns.policy_search(session, transitions=a.transitions, seed=seed, prior=prior,
                                             stabilization=a.stabilization)
            t0 = time.perf_counter()
            _b, ppath, pstats = PS.policy_search(session, deadline=None, transitions=a.transitions, seed=seed,
                                                 variant="gnrpa", prior=prior, stabilization=a.stabilization)
            pt = time.perf_counter() - t0
            pscore = int(_b.state.score)
            same = (tuple(ppath) == npath and pscore == nstats["best_score"]
                    and all(pstats[k] == nstats[k] for k in ("expansions", "rollouts", "incomplete_rollouts",
                                                              "adaptation_choices", "incumbent_updates",
                                                              "top_level_calls", "top_level_adaptations")))
            rec = {"round": list(key), "seed": seed, "transitions": a.transitions, "identical": same,
                   "python_score": pscore, "native_score": nstats["best_score"], "native_status": nstats["status"],
                   "python_seconds": round(pt, 2), "native_seconds": round(nstats["elapsed_seconds"], 3),
                   "speedup": round(pt / max(1e-9, nstats["elapsed_seconds"]), 1),
                   "python_stats": {k: pstats[k] for k in ("expansions", "rollouts", "incomplete_rollouts",
                                                           "adaptation_choices", "incumbent_updates",
                                                           "top_level_calls", "top_level_adaptations")},
                   "native_stats": {k: nstats[k] for k in ("expansions", "rollouts", "incomplete_rollouts",
                                                           "adaptation_choices", "incumbent_updates",
                                                           "top_level_calls", "top_level_adaptations")}}
            if not same:
                rec["first_path_diff"] = next((i for i, (x, y) in enumerate(zip(ppath, npath)) if tuple(x) != y),
                                              min(len(ppath), len(npath)))
            out.append(rec)
            print(json.dumps({k: rec[k] for k in ("round", "seed", "identical", "python_score", "native_score",
                                                  "native_status", "python_seconds", "native_seconds", "speedup")}),
                  flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("IDENTICAL", sum(r["identical"] for r in out), "/", len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
