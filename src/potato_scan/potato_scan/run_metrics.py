"""Per-phase timing and per-stage success, in the form this field reports it.

numpy-free, ROS-free, so it can be unit tested and so the same accounting can
be reused by a bench script as by the live nodes.

WHY THIS SHAPE

Agricultural-robotics papers report two things about a cell like this one,
and this project could produce neither. Both are conventions worth matching
exactly, because matching them is what makes a number comparable to a
published one rather than just a number.

  PHASE-BROKEN CYCLE TIME, not a single total. The tissue-sampling robot
  this project is closest to reports 10.4 s split into vision 2.1 s, grasp
  2.6 s, manipulation 5.7 s (approach+insert 1.9, transport 2.2, return 1.6)
  and end-effector 1.0 s. A total alone says the cell is slow; the split says
  which part to fix. A pineapple eye-removal machine in the same literature
  reports 156.7 s per fruit -- an order of magnitude apart, and only the
  breakdown explains why.

  SUCCESS DECOMPOSED BY STAGE, not one rate. The canonical harvesting review
  (50 systems) reports localization success, detachment success, overall
  success and damage rate separately, because a cell that localises well and
  detaches badly needs different work from one that does the reverse. Its
  averages -- localization 85%, detachment 75%, overall 66%, damage 5%,
  cycle 33 s -- are the yardstick a new cell gets held to.

So: localization is scored against a hand count of the eyes actually on the
potato (which is the only honest source of recall), and removal against what
the drill did, and the two are never merged into a single figure.
"""
import time


class PhaseTimer:
    """Accumulates wall time per named phase.

    A context manager rather than a pair of calls, so a phase cannot be left
    open by an early return or an exception -- which is exactly what happens
    on the paths worth timing, since those are the ones that abort.
    """

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def __call__(self, phase):
        return _Phase(self, phase)

    def record(self, phase, seconds):
        self.totals[phase] = self.totals.get(phase, 0.0) + float(seconds)
        self.counts[phase] = self.counts.get(phase, 0) + 1

    @property
    def total(self):
        return sum(self.totals.values())

    def mean(self, phase):
        count = self.counts.get(phase, 0)
        return (self.totals[phase] / count) if count else 0.0


class _Phase:
    def __init__(self, timer, phase):
        self.timer, self.phase = timer, phase

    def __enter__(self):
        self.started = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.timer.record(self.phase, time.monotonic() - self.started)
        return False       # never swallow: a failed phase still gets timed


class RunMetrics:
    """One potato's worth of accounting."""

    def __init__(self, timer=None):
        self.timer = timer or PhaseTimer()
        self.eyes_detected = 0
        self.eyes_present = None      # hand count; None until someone provides it
        self.attempts = []            # drill_task_planner.EyeAttempt
        self.damaged = 0

    def set_ground_truth(self, eyes_present):
        """The hand count of eyes actually on this potato.

        Recall cannot be computed without it, and nothing in the pipeline can
        supply it -- a detector cannot tell you what it failed to detect. Left
        unset, localization is reported as unknown rather than guessed.
        """
        self.eyes_present = int(eyes_present)

    @property
    def localization_rate(self):
        if not self.eyes_present:
            return None
        return min(self.eyes_detected / self.eyes_present, 1.0)

    @property
    def removal_rate(self):
        """Of the eyes that were found, how many were drilled to depth.
        Comparable to a complete-removal rate in the eye-removal literature."""
        if not self.attempts:
            return None
        return sum(1 for a in self.attempts if a.status == 'reached') / len(self.attempts)

    @property
    def overall_rate(self):
        """Detected AND removed, out of the eyes actually present -- the
        end-to-end figure, and always the lowest of the three."""
        if not self.eyes_present:
            return None
        removed = sum(1 for a in self.attempts if a.status == 'reached')
        return min(removed / self.eyes_present, 1.0)

    @property
    def damage_rate(self):
        if not self.attempts:
            return None
        return self.damaged / len(self.attempts)

    def format_report(self):
        def percent(value):
            return 'unknown' if value is None else f'{value * 100:.1f}%'

        lines = ['', 'run metrics:', '  success by stage:']
        if self.eyes_present is None:
            lines.append(f'    localization : {self.eyes_detected} detected, '
                         f'ground truth not provided -- call set_ground_truth()')
        else:
            lines.append(f'    localization : {percent(self.localization_rate)} '
                         f'({self.eyes_detected} detected / {self.eyes_present} present)')
        lines.append(f'    removal      : {percent(self.removal_rate)} '
                     f'(of {len(self.attempts)} attempted)')
        lines.append(f'    overall      : {percent(self.overall_rate)}')
        lines.append(f'    damage       : {percent(self.damage_rate)}')

        lines.append('  cycle time:')
        for phase in sorted(self.timer.totals, key=lambda p: -self.timer.totals[p]):
            total = self.timer.totals[phase]
            count = self.timer.counts[phase]
            share = 100.0 * total / self.timer.total if self.timer.total else 0.0
            lines.append(f'    {phase:<14} {total:>7.2f}s  ({share:>4.1f}%, '
                         f'{count} x {self.timer.mean(phase):.2f}s)')
        lines.append(f'    {"TOTAL":<14} {self.timer.total:>7.2f}s')
        return '\n'.join(lines)
