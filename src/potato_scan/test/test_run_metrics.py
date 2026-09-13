"""Checks the accounting: phase timing, the three success rates kept
separate, and that an unknown ground truth is reported as unknown rather
than quietly guessed."""
import time

import pytest

from potato_scan.drill_task_planner import EyeAttempt, depth_collar_spec
from potato_scan.run_metrics import PhaseTimer, RunMetrics


# --- phase timing --------------------------------------------------------

def test_phases_are_counted_and_summed_separately():
    timer = PhaseTimer()
    for _ in range(3):
        with timer('drill'):
            time.sleep(0.01)
    with timer('approach'):
        time.sleep(0.02)

    assert timer.counts['drill'] == 3
    assert timer.counts['approach'] == 1
    assert 0.025 < timer.totals['drill'] < 0.20
    assert timer.total == pytest.approx(timer.totals['drill'] + timer.totals['approach'])


def test_an_unrecorded_phase_has_a_zero_mean_rather_than_raising():
    assert PhaseTimer().mean('nonexistent') == 0.0


def test_a_phase_that_raises_is_still_timed_and_still_raises():
    """Otherwise the cycle-time report silently omits exactly the attempts
    that went wrong, which are the ones worth measuring."""
    timer = PhaseTimer()
    with pytest.raises(RuntimeError):
        with timer('drill'):
            raise RuntimeError('boom')
    assert timer.counts['drill'] == 1
    assert timer.totals['drill'] > 0.0


# --- the three rates -----------------------------------------------------

@pytest.fixture
def run():
    """4 of 6 eyes found, 2 of those 4 drilled to depth."""
    metrics = RunMetrics()
    metrics.eyes_detected = 4
    metrics.attempts = [EyeAttempt(0, 'reached'), EyeAttempt(1, 'reached'),
                        EyeAttempt(2, 'force_limit'), EyeAttempt(3, 'fixture_blocked')]
    return metrics


def test_removal_is_knowable_without_ground_truth_but_localization_is_not(run):
    assert run.localization_rate is None
    assert run.overall_rate is None
    assert run.removal_rate == pytest.approx(0.5)


def test_the_rates_separate_once_ground_truth_arrives(run):
    run.set_ground_truth(6)
    assert run.localization_rate == pytest.approx(4 / 6)
    assert run.removal_rate == pytest.approx(0.5)
    assert run.overall_rate == pytest.approx(2 / 6)


def test_end_to_end_can_never_beat_either_stage(run):
    run.set_ground_truth(6)
    assert run.overall_rate <= min(run.localization_rate, run.removal_rate)


def test_the_report_carries_all_three_and_the_damage_rate(run):
    run.set_ground_truth(6)
    run.timer.record('raster', 61.2)
    run.timer.counts['raster'] = 40
    run.timer.record('gap-filling', 18.4)
    run.timer.counts['gap-filling'] = 7
    run.timer.record('drill', 12.1)
    run.timer.counts['drill'] = 4
    run.damaged = 1

    report = run.format_report()
    assert 'localization : 66.7%' in report
    assert 'overall      : 33.3%' in report
    assert 'damage       : 25.0%' in report


def test_unknown_ground_truth_says_so():
    metrics = RunMetrics()
    metrics.eyes_detected = 3
    assert 'ground truth not provided' in metrics.format_report()


# --- the printed depth collar --------------------------------------------

def test_the_collar_face_sits_at_the_intended_depth():
    spec = depth_collar_spec(0.00325, 0.008, 40.0)
    assert spec['offset_from_tip_m'] == pytest.approx(0.008)


def test_the_collar_is_never_narrower_than_the_hole_it_must_stop_against():
    spec = depth_collar_spec(0.00325, 0.008, 40.0)
    assert spec['min_outer_diameter_m'] > 0.00325


def test_contact_pressure_sits_a_safety_factor_below_puncture():
    spec = depth_collar_spec(0.00325, 0.008, 40.0)
    assert (spec['pressure_at_max_force_pa'] * spec['safety_factor']
            == pytest.approx(spec['puncture_stress_pa'], abs=1.0))


@pytest.mark.parametrize('bit_m,force_n', [(0.00325, 90.0), (0.005, 40.0)])
def test_the_collar_grows_with_both_force_and_bit_diameter(bit_m, force_n):
    baseline = depth_collar_spec(0.00325, 0.008, 40.0)
    assert (depth_collar_spec(bit_m, 0.008, force_n)['min_outer_diameter_m']
            > baseline['min_outer_diameter_m'])
