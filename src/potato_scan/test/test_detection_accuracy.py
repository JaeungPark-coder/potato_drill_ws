"""Checks the scoring itself, since a scorer that flatters the detector is
worse than no scorer at all."""
import numpy as np
import pytest

from potato_scan.detection_accuracy import (
    angle_between_deg, format_report, match_detections, summarize)

TRUTH = np.array([[0.500, 0.000, 0.185],
                  [0.470, 0.020, 0.150],
                  [0.520, -0.015, 0.160]])


# --- the angle measure ---------------------------------------------------

@pytest.mark.parametrize('a,b,expected', [
    ([0, 0, 1], [0, 0, 1], 0.0),
    ([0, 0, 1], [0, 1, 0], 90.0),
    ([0, 0, 1], [0, 0, -1], 180.0),
])
def test_known_angles(a, b, expected):
    assert angle_between_deg(a, b)[0] == pytest.approx(expected, abs=1e-9)


def test_tiny_angles_stay_resolvable():
    """arccos of a dot product gives up around 1e-8 rad; these angles are
    routinely small and the small ones are the interesting ones."""
    tiny = np.radians(1e-4)
    got = angle_between_deg([0, 0, 1], [np.sin(tiny), 0, np.cos(tiny)])[0]
    assert got == pytest.approx(1e-4, rel=1e-6)


def test_the_measure_ignores_vector_length():
    assert angle_between_deg([0, 0, 5], [0, 0, 0.2])[0] == pytest.approx(0.0, abs=1e-9)


# --- matching ------------------------------------------------------------

def test_a_perfect_detection_matches_everything():
    result = match_detections(TRUTH, TRUTH)
    assert len(result['matches']) == 3
    assert result['missed'] == [] and result['spurious'] == []


def test_a_missed_eye_is_a_miss_not_a_bad_match():
    result = match_detections(TRUTH[:2], TRUTH)
    assert len(result['matches']) == 2
    assert result['missed'] == [2]
    assert result['spurious'] == []


def test_an_invented_eye_is_spurious():
    detected = np.vstack([TRUTH, [[0.40, 0.10, 0.30]]])
    result = match_detections(detected, TRUTH)
    assert result['spurious'] == [3]
    assert result['missed'] == []


def test_nothing_matches_beyond_the_tolerance():
    """Past some distance a detection is not a worse localisation of that
    eye, it is a different thing, and averaging it in would report a
    detection failure as a precision one."""
    detected = TRUTH + np.array([0.02, 0.0, 0.0])
    result = match_detections(detected, TRUTH, tolerance=0.008)
    assert result['matches'] == []
    assert len(result['missed']) == 3 and len(result['spurious']) == 3


def test_the_closest_pair_wins_rather_than_the_first_considered():
    """The failure greedy-by-truth makes: eye 0 is considered first and its
    nearest detection is the one that belongs to eye 1, which would read as
    one miss and one spurious where the truth is two good matches."""
    truth = np.array([[0.500, 0.0, 0.180], [0.500, 0.0, 0.186]])
    detected = np.array([[0.500, 0.0, 0.1855], [0.500, 0.0, 0.1805]])

    result = match_detections(detected, truth, tolerance=0.008)
    assert len(result['matches']) == 2
    assert result['missed'] == [] and result['spurious'] == []
    pairing = {t: d for t, d, _, _ in result['matches']}
    assert pairing == {0: 1, 1: 0}


def test_one_detection_cannot_satisfy_two_eyes():
    truth = np.array([[0.500, 0.0, 0.180], [0.500, 0.0, 0.184]])
    result = match_detections(np.array([[0.500, 0.0, 0.182]]), truth, tolerance=0.008)
    assert len(result['matches']) == 1
    assert len(result['missed']) == 1


# --- normals, the thing that actually failed ----------------------------

def test_a_wrong_normal_is_reported_even_when_the_position_is_perfect():
    """The 2026-09-14 case: candidates in plausible places whose normals were
    up to 84 degrees off, which no count of eyes found can express."""
    truth_normals = np.array([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
    detected_normals = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 0, 1.0]])

    result = match_detections(TRUTH, TRUTH, detected_normals=detected_normals,
                              truth_normals=truth_normals)
    s = summarize(result)
    assert s['found'] == 3
    assert s['position_worst_m'] == pytest.approx(0.0, abs=1e-12)
    assert s['normal_worst_deg'] == pytest.approx(90.0, abs=1e-6)


def test_normals_are_optional():
    result = match_detections(TRUTH, TRUTH)
    assert summarize(result)['normal_worst_deg'] is None


# --- summary contract ----------------------------------------------------

def test_errors_are_taken_over_matched_eyes_only():
    """A missed eye is a detection failure. Folding its distance into the
    position figure would report it as a localisation one."""
    far = np.vstack([TRUTH[:2], [[0.40, 0.10, 0.30]]])
    s = summarize(match_detections(far, TRUTH, tolerance=0.008))
    assert s['found'] == 2 and s['missed'] == 1
    assert s['position_worst_m'] == pytest.approx(0.0, abs=1e-12)


def test_nothing_matched_reports_unknown_rather_than_zero():
    s = summarize(match_detections(np.zeros((0, 3)), TRUTH))
    assert s['found'] == 0
    assert s['position_mean_m'] is None
    assert s['normal_mean_deg'] is None


def test_recall_is_none_when_there_is_no_truth():
    assert summarize(match_detections(TRUTH, np.zeros((0, 3))))['recall'] is None


# --- the printed report --------------------------------------------------

def test_the_report_names_every_eye_and_every_verdict():
    detected = np.vstack([TRUTH[:2], [[0.40, 0.10, 0.30]]])
    report = format_report(match_detections(detected, TRUTH, tolerance=0.008))
    assert 'NOT DETECTED' in report
    assert 'matches no eye' in report
    assert 'found        : 2/3' in report


def test_the_report_says_when_it_is_over_target():
    detected = TRUTH + np.array([0.005, 0.0, 0.0])      # 5 mm, target is 2 mm
    report = format_report(match_detections(detected, TRUTH, tolerance=0.008))
    assert 'OVER TARGET' in report


def test_the_report_survives_having_nothing_to_report():
    assert 'nothing to measure' in format_report(
        match_detections(np.zeros((0, 3)), TRUTH))
