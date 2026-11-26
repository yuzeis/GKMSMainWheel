# Copyright (C) 2026 花吹雪また一年P
# SPDX-License-Identifier: AGPL-3.0-or-later
"""the route search's switches read in one place, its backend registered once, and what one search call is given."""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Optional


# The GNRPA nesting depth a search takes when its caller gives none (levels=None), by budget.  Measured with
# GakumasCLI/native/search_ab.py, level 3 against level 2 at the same budget, game by game (wins / losses over the six
# coverage baselines x 50 exams; the native core, one search seed): a third level loses on small budgets
# (40,000 transitions 63 / 236, 120,000 87 / 213), breaks even at 240,000 (153 / 143) and wins from 320,000 on
# (185 / 111; 480,000 219 / 77; 1,920,000 244 / 53), for every archetype at 480,000 and above.  By seconds: 0.5 s
# 195 / 101, 1 s 228 / 69, 2 s 238 / 59 on an M-series Mac with 8 searches at once (0.5 s there is ~0.35M-0.39M
# transitions, 1 s ~0.7M) -- the seconds threshold is set at 1 s so that a machine half as fast still clears 320,000.
AUTO_LEVELS_MIN_TRANSITIONS = 320_000
AUTO_LEVELS_MIN_DEADLINE = 1.0


def auto_levels(transitions=None, deadline=None):
    """The nesting depth for a budget: 3 when every budget given is at or above its threshold, else 2."""
    fits = []
    if transitions is not None:
        fits.append(int(transitions) >= AUTO_LEVELS_MIN_TRANSITIONS)
    if deadline is not None:
        fits.append(float(deadline) >= AUTO_LEVELS_MIN_DEADLINE)
    return 3 if fits and all(fits) else 2


def _text(env, name, default=""):
    return env.get(name, default).strip().lower()


def _int_or_none(env, name):
    raw = env.get(name)
    return int(raw) if raw else None


def _float_or_none(env, name):
    raw = env.get(name)
    return float(raw) if raw else None


@dataclasses.dataclass(frozen=True)
class SearchSwitches:
    run_transitions: int = 500000
    helper_feed: str = "sync"
    er_cut: str = ""
    er_root: str = ""
    levels: Optional[int] = None
    iterations: Optional[int] = None
    tail_polish: str = "native"
    er_job_seconds: Optional[float] = None
    er_tail_seconds: Optional[float] = None
    tail_max_turns: int = 5
    tail_node_cap: int = 200000000
    tail_threads: int = 0
    prefix_four: bool = False
    root_strata: bool = False
    elite_prior: bool = False
    deep_nesting: bool = False
    archive_prior: str = ""
    warm_suffix: bool = False
    pick_variants: int = 0

    @property
    def chunk_experiment(self):
        return any((self.prefix_four, self.root_strata, self.elite_prior, self.deep_nesting,
                    self.archive_prior, self.warm_suffix))

    @classmethod
    def from_env(cls, env=None) -> "SearchSwitches":
        env = os.environ if env is None else env
        return cls(
            run_transitions=int(env.get("GKMS_RUN_TRANSITIONS", "500000") or 0),
            helper_feed=_text(env, "GKMS_HELPER_FEED", "sync") or "sync",
            er_cut=_text(env, "GKMS_ER_CUT"),
            er_root=_text(env, "GKMS_ER_ROOT"),
            levels=_int_or_none(env, "GKMS_NATIVE_LEVELS"),
            iterations=_int_or_none(env, "GKMS_NATIVE_ITERATIONS"),
            tail_polish=env.get("GKMS_NATIVE_TAIL_POLISH", "native"),
            er_job_seconds=_float_or_none(env, "GKMS_NATIVE_ER_JOB_SECONDS"),
            er_tail_seconds=_float_or_none(env, "GKMS_NATIVE_ER_TAIL_SECONDS"),
            tail_max_turns=int(env.get("GKMS_NATIVE_TAIL_MAX_TURNS", "5")),
            tail_node_cap=int(env.get("GKMS_NATIVE_TAIL_NODE_CAP", "200000000")),
            tail_threads=int(env.get("GKMS_TAIL_THREADS") or 0),
            prefix_four=env.get("GKMS_PREFIX_FOUR") == "1",
            root_strata=env.get("GKMS_ROOT_STRATA") == "1",
            elite_prior=env.get("GKMS_ELITE_PRIOR") == "1",
            deep_nesting=env.get("GKMS_DEEP_NESTING") == "1",
            archive_prior=env.get("GKMS_ARCHIVE_PRIOR", ""),
            warm_suffix=env.get("GKMS_WARM_SUFFIX") == "1",
            pick_variants=int(env.get("GKMS_PICK_VARIANTS") or 0))

    def tail_thread_count(self, cpus=None) -> int:
        """hif_session.TAIL_THREADS: the switch, else the machine's cores minus two, 1..16."""
        return max(1, min(16, self.tail_threads or (cpus if cpus is not None else (os.cpu_count() or 2)) - 2))


@dataclasses.dataclass(frozen=True)
class SearchBackend:
    name: str
    policy_search: Callable[..., Any]
    pool_worker: Optional[Callable[..., Any]] = None
    tail_polish: Optional[Callable[..., Any]] = None
    er_job_seconds: Optional[float] = None
    er_tail_seconds: Optional[float] = None


_ACTIVE: dict = {}


def register(backend: SearchBackend) -> SearchBackend:
    """Install ``backend`` into the search modules of this process (idempotent for the same backend)."""
    from GakumasCLI.scenario import policy_search as PS
    from GakumasCLI.scenario import hif_route_search as HRS
    PS.policy_search = backend.policy_search
    HRS.policy_search = backend.policy_search
    if backend.pool_worker is not None:
        HRS._pool_worker = backend.pool_worker
    if backend.tail_polish is not None and hasattr(HRS, "TAIL_POLISH"):
        HRS.TAIL_POLISH = backend.tail_polish
    if backend.er_job_seconds is not None:
        HRS.ER_JOB_SECONDS = float(backend.er_job_seconds)
    if backend.er_tail_seconds is not None:
        HRS.ER_JOB_TAIL_SECONDS = float(backend.er_tail_seconds)
    _ACTIVE["backend"] = backend
    return backend


def active() -> Optional[SearchBackend]:
    """The backend registered in this process (None: the Python search of the modules as imported)."""
    return _ACTIVE.get("backend")


@dataclasses.dataclass(frozen=True)
class SearchContext:
    target: Optional[int] = None
    budget: Optional[float] = None
    cancel: Optional[Callable[[], bool]] = None
    workspace: Optional[str] = None
    prior: Any = None
    switches: SearchSwitches = dataclasses.field(default_factory=SearchSwitches.from_env)

    @classmethod
    def here(cls, context: Optional["SearchContext"] = None, **fields) -> "SearchContext":
        """``context``, else one with the switches read now."""
        if context is not None:
            return dataclasses.replace(context, **fields) if fields else context
        return cls(**fields)
