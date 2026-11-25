# Copyright (C) 2026 花吹雪また一年P
# SPDX-License-Identifier: AGPL-3.0-or-later
"""run the main-line route search with the native GNRPA.

      python -B GakumasCLI/gkms.py native.route_search_native --engine <repo> -- <hif_route_search.py arguments>
    Python 2-turn polish (deepening) GKMS_NATIVE_TAIL_MAX_TURNS deepest native polish
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_NATIVE = {}
_ORIG = {}


def _master_dir():
    """GKMS_NATIVE_MASTER, else the engine's own master tables."""
    env = os.environ.get("GKMS_NATIVE_MASTER")
    if env:
        return Path(env)
    from GakumasCLI.paths import MASTER_DIR
    return MASTER_DIR


def _native(master=None):
    """The process's native search; ``master``: an already loaded Master of ``_master_dir``."""
    if "ns" not in _NATIVE:
        import json
        from GakumasCLI.scenario.exam_session import Master
        from GakumasCLI.native.diffrun import Native
        from GakumasCLI.native.native_search import NativeSearch
        lib = os.environ["GKMS_NATIVE_LIB"]
        strings = json.load(open(os.environ["GKMS_NATIVE_STRINGS"], encoding="utf-8"))
        nat = Native(lib)
        nat.load_master(master if master is not None else Master.load(_master_dir()), strings)
        _NATIVE["ns"] = NativeSearch(nat)
    return _NATIVE["ns"]


def _setup_worker(master, master_path):
    """hif_route_search.WORKER_SETUP in a native pool worker -- the core is loaded on the master the worker just loaded,
    before it reports ready.
    """
    if not (os.environ.get("GKMS_NATIVE_LIB") and os.environ.get("GKMS_NATIVE_STRINGS")):
        return
    same = Path(master_path).resolve() == _master_dir().resolve()
    _native(master if same else None)


def native_policy_search(session, *, deadline=5, seed=182, variant="gnrpa", transitions=None, levels=None,
                         iterations=8, on_incumbent=None, prior=None, cancel=None, initial_route=None,
                         persist_policy=True, repetition_limit=0, stabilization=1, rules=None):
    """``rules``: a timeline's RuleSet -- on a core that takes rules the native search applies them"""
    py = _ORIG["policy_search"]
    kwargs = dict(deadline=deadline, seed=seed, variant=variant, transitions=transitions, levels=levels,
                  iterations=iterations, on_incumbent=on_incumbent, prior=prior, cancel=cancel,
                  initial_route=initial_route, persist_policy=persist_policy,
                  repetition_limit=repetition_limit, stabilization=stabilization)
    if rules is not None:
        kwargs["rules"] = rules
    from GakumasCLI.scenario.search_context import SearchSwitches
    switches = SearchSwitches.from_env()
    if switches.levels:
        levels = switches.levels
        kwargs["levels"] = levels
    if switches.iterations:
        iterations = switches.iterations
        kwargs["iterations"] = iterations
    if initial_route is not None or on_incumbent is not None or cancel is not None:
        b, path, stats = py(session, **kwargs)
        stats["native_fallback"] = "argument"
        return b, path, stats
    from GakumasCLI.scenario import solver as SV
    try:
        extra = {"rules": rules} if rules is not None else {}
        if switches.pick_variants > 0:
            extra["pick_variants"] = switches.pick_variants
        path, stats = _native().policy_search(session, deadline=deadline, seed=seed, variant=variant,
                                              transitions=transitions, levels=levels, iterations=iterations,
                                              prior=prior, stabilization=stabilization,
                                              repetition_limit=repetition_limit, persist_policy=persist_policy,
                                              **extra)
    except Exception as exc:   # noqa: BLE001
        b, path, stats = py(session, **kwargs)
        stats["native_fallback"] = f"exception: {type(exc).__name__}: {str(exc)[:160]}"
        return b, path, stats
    if stats["status"] != 0 or not path:
        b, path2, stats2 = py(session, **kwargs)
        stats2["native_fallback"] = f"status {stats['status']} line {stats['unsup_line']} info {stats['unsup_info']}"
        return b, path2, stats2
    stats["native"] = True
    stats["native_levels"], stats["native_iterations"] = stats.get("levels", levels), iterations
    return SV.replay_session(session, path), path, stats


def native_tail_polish(session, path, turns, *, deadline=5.0, cancel=None, rules=None, max_turns=None):
    """the search job's closing tail polish on the native core, deepening while it fits
    (GakumasCLI.native.tail_native.polish); the Python polish takes any state the native core refuses.
    """
    from GakumasCLI.native import tail_native as TN
    from GakumasCLI.scenario import solver as SV
    extra = {}
    if rules is not None:
        extra["rules"] = rules
    if max_turns is not None:
        extra["max_turns"] = int(max_turns)
    try:
        ns = _native()
    except Exception:   # noqa: BLE001
        if rules is not None:
            return tuple(path), {"skipped": "rules", "native": False}
        path, meta = SV.exact_tail_polish(session, path, turns, deadline=deadline, cancel=cancel)
        return path, dict(meta or {}, native=False)
    try:
        return TN.polish(session, path, turns, deadline=deadline, cancel=cancel, nat=ns.nat,
                         fallback=None if rules is not None else SV.exact_tail_polish, **extra)
    finally:
        ns._cache = None


def backend(switches=None):
    """the native search backend (scenario.search_context.SearchBackend) with this process's switches."""
    from GakumasCLI.scenario.search_context import SearchBackend, SearchSwitches
    switches = switches or SearchSwitches.from_env()
    return SearchBackend(name="native", policy_search=native_policy_search, pool_worker=_pool_worker,
                         tail_polish=native_tail_polish if switches.tail_polish != "python" else None,
                         er_job_seconds=switches.er_job_seconds, er_tail_seconds=switches.er_tail_seconds)


def install():
    """Register the native backend in this process (idempotent)."""
    from GakumasCLI.scenario import policy_search as PS
    from GakumasCLI.scenario import hif_route_search as HRS
    from GakumasCLI.scenario import search_context as SC
    if "policy_search" not in _ORIG:
        _ORIG["policy_search"] = PS.policy_search
        _ORIG["pool_worker"] = HRS._pool_worker
    SC.register(backend())


def _pool_worker(connection, prior_path, master_dir=None, cancel=None):
    """SeedPool target: install the native backend in the spawned worker, then run the original worker loop."""
    install()
    from GakumasCLI.scenario import hif_route_search as HRS
    HRS.WORKER_SETUP = _setup_worker
    return _ORIG["pool_worker"](connection, prior_path, master_dir, cancel)


def main(argv=None):
    """python -B GakumasCLI/gkms.py native.route_search_native --engine <repo> [--levels N] [--iterations N]
           [--er-job-seconds S] [--er-tail-seconds S] -- <hif_route_search.py arguments>
    """
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--levels", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--er-job-seconds", type=float, default=None)
    ap.add_argument("--er-tail-seconds", type=float, default=None)
    own = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    for name, value in (("GKMS_NATIVE_LEVELS", own.levels), ("GKMS_NATIVE_ITERATIONS", own.iterations),
                        ("GKMS_NATIVE_ER_JOB_SECONDS", own.er_job_seconds),
                        ("GKMS_NATIVE_ER_TAIL_SECONDS", own.er_tail_seconds)):
        if value is not None:
            os.environ[name] = str(value)
    engine = str(Path(own.engine).resolve())
    os.environ["GKMS_NATIVE_LIB"] = str(Path(os.environ["GKMS_NATIVE_LIB"]).resolve())
    os.environ["GKMS_NATIVE_STRINGS"] = str(Path(os.environ["GKMS_NATIVE_STRINGS"]).resolve())
    os.chdir(engine)
    install()
    from GakumasCLI.scenario import hif_route_search as HRS
    print("native shape:", {k: os.environ.get(k) for k in ("GKMS_NATIVE_LEVELS", "GKMS_NATIVE_ITERATIONS",
                                                            "GKMS_NATIVE_ER_JOB_SECONDS",
                                                            "GKMS_NATIVE_ER_TAIL_SECONDS")},
          "ER job", HRS.ER_JOB_SECONDS, "+", HRS.ER_JOB_TAIL_SECONDS, flush=True)
    return HRS.main(rest)


if __name__ == "__main__":
    sys.exit(main())
