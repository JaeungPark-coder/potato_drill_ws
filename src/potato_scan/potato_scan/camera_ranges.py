"""Will this camera see anything at the distance the scan actually uses?

numpy-free, ROS-free. Exists because a depth camera below its minimum range
does not degrade gracefully -- depth breaks up and then disappears -- and the
failure is invisible in every log: the cloud simply arrives empty or full of
holes, which reads as occlusion, a bad viewpoint, or a potato that did not
scan well.

THE DISTANCE THAT MATTERS IS NOT scan_radius

scan_radius is measured from the potato's CENTRE, and the camera images its
SURFACE, which is a potato-radius nearer. At the configured 0.15 m radius
around a 35 mm potato the nearest surface sits 0.115 m from the lens -- 23%
closer than the number in the config file, and on the wrong side of the
minimum range for several common cameras.

THE NUMBERS

Minimum ranges differ by nearly an order of magnitude across cameras that
otherwise look interchangeable, and only one common model is built for this
distance:

    D405     0.07 m    purpose-built for close range, ideal 0.07-0.50 m
    D435     0.168 m   at 848x480; minimum scales with X resolution, so a
                       higher-resolution mode pushes it further out
    D415     ~0.45 m
    D455     0.40 m    a long-range camera; unusable this close
    Gemini335 0.10 m   usable, but its optimal band starts at 0.26 m

Taken from the manufacturers' own documentation via a literature review
rather than measured here, and resolution-dependent for the stereo models --
so this is a screen that catches an obviously wrong choice, not a substitute
for a bench check at the working distance and resolution actually in use.
"""

# name -> (absolute minimum depth in m, start of the optimal band in m, note)
CAMERA_MIN_RANGE_M = {
    'd405': (0.07, 0.07, 'purpose-built for close range (ideal 0.07-0.50 m)'),
    'd435': (0.168, 0.30, 'minimum quoted at 848x480 and scales with X resolution'),
    'd435i': (0.168, 0.30, 'minimum quoted at 848x480 and scales with X resolution'),
    'd415': (0.45, 0.50, 'long minimum range'),
    'd455': (0.40, 0.60, 'a long-range camera; not intended for close work'),
    'gemini335': (0.10, 0.26, 'usable below its optimal band, with degraded quality'),
}


def nearest_surface_distance_m(scan_radius_m, potato_radius_m):
    """How far the lens actually is from the nearest surface it images."""
    return float(scan_radius_m) - float(potato_radius_m)


def check(camera_model, scan_radius_m, potato_radius_m):
    """Returns (ok, message). ok is False only for a camera known to be
    unable to focus this close -- an unrecognised model returns True with a
    message saying it could not be checked, since refusing to run over a
    name this table has not heard of would be worse than saying so.
    """
    distance = nearest_surface_distance_m(scan_radius_m, potato_radius_m)
    key = str(camera_model).strip().lower().replace('-', '').replace('_', '')
    key = key.replace('realsense', '').replace('intel', '').replace('orbbec', '').strip()

    if key not in CAMERA_MIN_RANGE_M:
        return True, (
            f'camera_model {camera_model!r} is not in the table, so its minimum range '
            f'was not checked. The nearest surface sits {distance * 100:.1f} cm from '
            f'the lens at scan_radius {scan_radius_m * 100:.0f} cm -- confirm that is '
            f'above the minimum for your model AND resolution before collecting.')

    minimum, optimal, note = CAMERA_MIN_RANGE_M[key]
    if distance < minimum:
        return False, (
            f'{camera_model} cannot focus this close: the nearest potato surface is '
            f'{distance * 100:.1f} cm away (scan_radius {scan_radius_m * 100:.0f} cm '
            f'minus a {potato_radius_m * 100:.0f} cm potato) but its minimum is '
            f'{minimum * 100:.1f} cm -- {note}. Depth will break up and vanish rather '
            f'than degrade, so the scan will look occluded instead of out of range. '
            f'Raise scan_radius above {minimum + potato_radius_m:.3f} m or use a '
            f'close-range camera.')
    if distance < optimal:
        return True, (
            f'{camera_model} will work at {distance * 100:.1f} cm but is below its '
            f'optimal band (starts at {optimal * 100:.1f} cm) -- {note}. Expect more '
            f'noise and dropout than the datasheet accuracy figures suggest.')
    return True, (f'{camera_model} at {distance * 100:.1f} cm is within its optimal '
                  f'band -- {note}.')
