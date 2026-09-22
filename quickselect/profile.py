"""Timing hooks.

Every stage of a segmentation pass reports how long it took, so the UI can show
where the time actually goes instead of a single opaque number. The overhead is
one ``perf_counter`` pair per stage -- well under a microsecond, so this stays
on in production rather than being a debug-only path.

Note on the module name: this shadows the standard library's ``profile`` only
for code inside this package that writes ``import profile``. Nothing here does;
all imports are relative (``from .profile import ...``), and under Python 3's
absolute-import rules an unrelated ``import profile`` elsewhere still finds the
stdlib module.

Usage::

    prof = Profiler()
    with prof.stage(GRAPH_BUILD):
        build_graph()
    print(prof.summary())
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

# Canonical stage names.  Keeping them as constants means a typo shows up as a
# NameError rather than as a silently separate row in the report.
COLOR_MODEL = "color_model"
GRAPH_BUILD = "graph_build"
MAXFLOW = "maxflow"
POSTPROCESS = "postprocess"
REFINE = "refine"
UPSAMPLE = "upsample"
CONSTRAINTS = "constraints"
PREPARE = "prepare"

STAGE_ORDER: Tuple[str, ...] = (
    PREPARE,
    CONSTRAINTS,
    COLOR_MODEL,
    GRAPH_BUILD,
    MAXFLOW,
    POSTPROCESS,
    UPSAMPLE,
    REFINE,
)


@dataclass
class Profiler:
    """Accumulates per-stage timings for one pass, plus a rolling history."""

    #: Milliseconds spent in each stage during the current pass.
    stages: Dict[str, float] = field(default_factory=dict)
    #: Free-form notes a stage wants to surface (node counts, ROI size, ...).
    notes: Dict[str, object] = field(default_factory=dict)
    #: Completed passes, newest last.  Bounded so a long session cannot grow it.
    history: List[Dict[str, float]] = field(default_factory=list)
    history_limit: int = 200
    enabled: bool = True

    _t0: float = 0.0

    # ------------------------------------------------------------------ #
    def begin(self) -> None:
        """Start a new pass, discarding whatever the previous one recorded."""
        self.stages = {}
        self.notes = {}
        self._t0 = time.perf_counter()

    def finish(self) -> Dict[str, float]:
        """Close the pass and push it onto the history."""
        total = (time.perf_counter() - self._t0) * 1000.0
        self.stages["total"] = total
        if self.enabled:
            self.history.append(dict(self.stages))
            if len(self.history) > self.history_limit:
                del self.history[: len(self.history) - self.history_limit]
        return self.stages

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and add it to ``stages`` (accumulating on repeats)."""
        if not self.enabled:
            yield
            return
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (
                time.perf_counter() - t
            ) * 1000.0

    def note(self, key: str, value: object) -> None:
        self.notes[key] = value

    def add(self, name: str, milliseconds: float) -> None:
        """Record a stage timed elsewhere (e.g. inside another Profiler)."""
        self.stages[name] = self.stages.get(name, 0.0) + milliseconds

    # ------------------------------------------------------------------ #
    @property
    def total(self) -> float:
        return self.stages.get("total", 0.0)

    def summary(self, width: int = 0) -> str:
        """One-line breakdown, ordered by pipeline stage, not by cost.

        Reading it in pipeline order makes it obvious *where* a regression
        appeared; sorting by cost hides that.
        """
        parts = []
        for name in STAGE_ORDER:
            if name in self.stages:
                parts.append(f"{name} {self.stages[name]:.1f}")
        for name, value in self.stages.items():
            if name not in STAGE_ORDER and name != "total":
                parts.append(f"{name} {value:.1f}")
        line = f"total {self.total:.1f} ms  =  " + " + ".join(parts)
        if width and len(line) > width:
            line = line[: width - 1] + "…"
        return line

    def rows(self) -> List[Tuple[str, float]]:
        """Stage/milliseconds pairs in pipeline order, for a table view."""
        out = [(n, self.stages[n]) for n in STAGE_ORDER if n in self.stages]
        out += [
            (n, v)
            for n, v in self.stages.items()
            if n not in STAGE_ORDER and n != "total"
        ]
        out.append(("total", self.total))
        return out

    def mean(self, name: str = "total", last: int = 20) -> float:
        """Average of the last ``last`` passes, for a stable status readout."""
        if not self.history:
            return 0.0
        window = self.history[-last:]
        values = [h.get(name, 0.0) for h in window]
        return sum(values) / len(values) if values else 0.0

    def percentile(self, pct: float, name: str = "total", last: int = 100) -> float:
        if not self.history:
            return 0.0
        values = sorted(h.get(name, 0.0) for h in self.history[-last:])
        if not values:
            return 0.0
        k = min(len(values) - 1, max(0, int(round(pct / 100.0 * (len(values) - 1)))))
        return values[k]

    def reset_history(self) -> None:
        self.history.clear()


#: A module-level profiler for callers that do not want to thread one through.
default_profiler = Profiler()


@contextmanager
def stage(name: str, profiler: Optional[Profiler] = None) -> Iterator[None]:
    """Time a block against ``profiler``, or the module-level one."""
    target = profiler if profiler is not None else default_profiler
    with target.stage(name):
        yield


class NullProfiler(Profiler):
    """Drop-in that records nothing, for the hot paths in batch scripts."""

    def __init__(self) -> None:
        super().__init__(enabled=False)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        yield

    def begin(self) -> None:
        self.stages = {}

    def finish(self) -> Dict[str, float]:
        return self.stages
