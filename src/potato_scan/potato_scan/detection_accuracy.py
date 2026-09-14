"""Score detected eyes against known ones.

Until now the only way to find out whether a detected eye was in the right
PLACE was step 5 of the bring-up order: drive the tip to it on a real robot
and measure with callipers. That is the right final answer and it is not
going away -- but it needs hardware, and it was blocking a question the
simulator can already answer, because isaac_sim_common.make_potato_mesh has
always returned the world-space position and outward normal of every pit it
carves. isaac_scene.py simply threw them away.

With them published, this module turns two sets of eyes into the numbers
that say how the detector actually did:

    localisation   how far each matched eye is from the truth
    orientation    how far its normal is from the true outward direction
    detection      how many were found, missed, or invented

Those are three separate failures and they are worth keeping separate. A
2026-09-14 Isaac Sim run found six plausible-looking candidates and then
could not drill two of them, because their NORMALS were up to 84 degrees
wrong while their positions may well have been fine -- a distinction no
count of "eyes found" can express, and the one that decided whether the bug
was in the detector or in the arm.

Pure geometry: no ROS, no Isaac, no robot. detection_accuracy_check is the
node that feeds it.
"""
import numpy as np


def _unit(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def angle_between_deg(a, b):
    """Angle between two directions, in degrees.

    Via arcsin of half the chord rather than arccos of the dot product:
    arccos is ill-conditioned near 0 degrees, where its derivative is
    infinite and float64 gives up at around 1e-8 rad. These angles are
    routinely small, and the difference between 0.001 and 0.02 degrees is
    exactly the regime a normal-quality question lives in.
    """
    a, b = _unit(np.atleast_2d(a)), _unit(np.atleast_2d(b))
    chord = np.linalg.norm(a - b, axis=-1)
    return np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


def match_detections(detected_positions, truth_positions, tolerance=0.008,
                     detected_normals=None, truth_normals=None):
    """Pair detections with true eyes, closest pair first.

    Globally closest-first rather than truth-by-truth: taking each true eye's
    nearest detection in turn lets whichever eye happens to be considered
    first claim a detection that sits much closer to another, which then
    reads as one miss and one spurious detection where the truth was two
    good matches. Sorting every pair by distance and consuming greedily is
    not optimal assignment, but it cannot make that particular mistake, and
    at the handful of eyes a potato carries the two agree.

    A pair further apart than `tolerance` is never matched: past some
    distance a "detection" is not a worse localisation of that eye, it is a
    different thing entirely, and averaging its error into the localisation
    figure would report a detection failure as a precision one.

    Returns a dict with `matches` [(truth index, detection index, distance,
    normal error degrees or None)], `missed` and `spurious` index lists.
    """
    detected_positions = np.asarray(detected_positions, dtype=float).reshape(-1, 3)
    truth_positions = np.asarray(truth_positions, dtype=float).reshape(-1, 3)

    pairs = []
    for t in range(len(truth_positions)):
        for d in range(len(detected_positions)):
            distance = float(np.linalg.norm(truth_positions[t] - detected_positions[d]))
            if distance <= tolerance:
                pairs.append((distance, t, d))
    pairs.sort()

    matches, used_truth, used_detection = [], set(), set()
    for distance, t, d in pairs:
        if t in used_truth or d in used_detection:
            continue
        used_truth.add(t)
        used_detection.add(d)

        normal_error = None
        if detected_normals is not None and truth_normals is not None:
            normal_error = float(angle_between_deg(
                np.asarray(detected_normals, dtype=float).reshape(-1, 3)[d],
                np.asarray(truth_normals, dtype=float).reshape(-1, 3)[t])[0])
        matches.append((t, d, distance, normal_error))

    matches.sort()
    return {
        'matches': matches,
        'missed': [t for t in range(len(truth_positions)) if t not in used_truth],
        'spurious': [d for d in range(len(detected_positions)) if d not in used_detection],
        'n_truth': len(truth_positions),
        'n_detected': len(detected_positions),
        'tolerance': float(tolerance),
    }


def summarize(result):
    """Headline numbers, with unknowns left as None rather than filled in."""
    distances = [distance for _, _, distance, _ in result['matches']]
    normal_errors = [error for _, _, _, error in result['matches'] if error is not None]

    return {
        'found': len(result['matches']),
        'of': result['n_truth'],
        'missed': len(result['missed']),
        'spurious': len(result['spurious']),
        # over MATCHED eyes only: the distance to an eye that was never found
        # is a detection failure, and folding it in here would report it as a
        # localisation one
        'position_mean_m': float(np.mean(distances)) if distances else None,
        'position_worst_m': float(np.max(distances)) if distances else None,
        'normal_mean_deg': float(np.mean(normal_errors)) if normal_errors else None,
        'normal_worst_deg': float(np.max(normal_errors)) if normal_errors else None,
        'recall': (len(result['matches']) / result['n_truth']
                   if result['n_truth'] else None),
    }


# Step 5's acceptance number, reused here so the simulator is scored against
# the same bar the callipers will be: calliper_check asks for a mean under
# about 2 mm. The simulator has no camera noise, so clearing it here is
# necessary and nowhere near sufficient.
POSITION_TARGET_M = 0.002

# An approach built from a normal much further off than this starts asking
# the arm to insert along the surface rather than into it. 45 degrees is the
# angle drill_controller already warns at, kept identical on purpose.
NORMAL_WARN_DEG = 45.0


def format_report(result, position_target_m=POSITION_TARGET_M,
                  normal_warn_deg=NORMAL_WARN_DEG):
    s = summarize(result)
    lines = [
        'detection accuracy vs ground truth',
        f"  found        : {s['found']}/{s['of']}"
        + (f"  (recall {s['recall'] * 100:.0f}%)" if s['recall'] is not None else ''),
        f"  missed       : {s['missed']}",
        f"  spurious     : {s['spurious']}"
        f"   (detections matching no eye within {result['tolerance'] * 1000:.0f}mm)",
    ]

    if s['position_mean_m'] is None:
        lines.append('  position     : nothing matched, so nothing to measure')
    else:
        verdict = 'ok' if s['position_mean_m'] <= position_target_m else 'OVER TARGET'
        lines.append(f"  position     : mean {s['position_mean_m'] * 1000:.2f}mm "
                     f"worst {s['position_worst_m'] * 1000:.2f}mm "
                     f"(target {position_target_m * 1000:.0f}mm -- {verdict})")

    if s['normal_mean_deg'] is not None:
        verdict = 'ok' if s['normal_worst_deg'] <= normal_warn_deg else 'OVER WARN'
        lines.append(f"  normal       : mean {s['normal_mean_deg']:.1f}deg "
                     f"worst {s['normal_worst_deg']:.1f}deg "
                     f"(warn {normal_warn_deg:.0f}deg -- {verdict})")

    lines.append('')
    for t, d, distance, normal_error in result['matches']:
        normal_text = '' if normal_error is None else f"  normal {normal_error:5.1f}deg"
        lines.append(f'    eye {t} <- detection {d}: '
                     f'{distance * 1000:5.2f}mm{normal_text}')
    for t in result['missed']:
        lines.append(f'    eye {t}: NOT DETECTED')
    for d in result['spurious']:
        lines.append(f'    detection {d}: matches no eye')

    return '\n'.join(lines)
