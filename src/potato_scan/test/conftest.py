"""Puts the package on sys.path so these tests run with no ROS 2 install.

Everything in this directory is deliberately importable without rclpy, without
a camera and without a robot: the point of the suite is to separate "my
install is wrong" from "my hardware is wrong" before anything is plugged in.
Nodes themselves are not imported -- the geometry, the planning and the
metrics were factored out of them into pure modules precisely so they could
be checked here.

    python -m pytest test/ -v        # from src/potato_scan
"""
import sys
from pathlib import Path

# src/potato_scan/, the directory holding the potato_scan python package
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
