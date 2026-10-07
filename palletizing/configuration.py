"""Shared configuration validation for both simulation backends."""

from pathlib import Path
import numpy as np
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        c = yaml.safe_load(stream)
    for section in (
        "simulation",
        "robot",
        "camera",
        "box",
        "table",
        "pallet",
        "process",
    ):
        if not isinstance(c.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

    def vector(section, key, length, positive=False):
        v = np.asarray(c[section][key], float)
        if (
            v.shape != (length,)
            or not np.isfinite(v).all()
            or (positive and np.any(v <= 0))
        ):
            raise ValueError(f"Invalid {section}.{key}")
        return v

    for section, key, length in (
        ("robot", "scan_position", 3),
        ("camera", "local_translation", 3),
        ("table", "center_xy", 2),
        ("pallet", "center_xy", 2),
    ):
        vector(section, key, length)
    for section, key, length in (
        ("box", "dimensions", 3),
        ("table", "dimensions", 3),
        ("pallet", "dimensions", 3),
        ("pallet", "spacing_xy", 2),
        ("robot", "max_joint_speed", 6),
    ):
        vector(section, key, length, True)
    for section, key in (
        ("robot", "scan_orientation"),
        ("camera", "local_orientation_ros"),
    ):
        if not np.isclose(np.linalg.norm(vector(section, key, 4)), 1.0, atol=1e-5):
            raise ValueError(f"{section}.{key} must be a unit wxyz quaternion")
    for section, keys in {
        "simulation": ("physics_hz", "render_hz"),
        "camera": ("fps", "min_points", "stable_frames"),
        "pallet": ("rows", "columns"),
    }.items():
        for key in keys:
            if type(c[section][key]) is not int or c[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for section, keys in {
        "simulation": ("max_seconds",),
        "robot": (
            "tool_length",
            "travel_z",
            "max_joint_acceleration",
            "max_linear_speed",
            "max_angular_speed",
            "kp",
            "kd",
            "position_tolerance",
            "orientation_tolerance",
            "waypoint_timeout",
        ),
        "camera": (
            "focal_length",
            "horizontal_aperture",
            "dimension_tolerance",
            "detection_timeout",
        ),
        "box": ("mass",),
        "process": (
            "approach_clearance",
            "release_gap",
            "grip_dwell",
            "release_dwell",
            "placement_tolerance",
            "break_force",
            "break_torque",
        ),
    }.items():
        for key in keys:
            value = float(c[section][key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} must be finite and positive")
    width, height = c["camera"]["resolution"]
    if any(type(v) is not int or v < 16 for v in (width, height)):
        raise ValueError("Camera resolution must contain two integers >=16")
    near, far = vector("camera", "clipping_range", 2, True)
    if near >= far:
        raise ValueError("Camera near plane must precede far plane")
    s = c["simulation"]
    if s["streaming"] not in ("webrtc", "native", "none"):
        raise ValueError("Streaming must be webrtc, native or none")
    for key in ("headless", "keep_open", "ros2_bridge"):
        if type(s[key]) is not bool:
            raise ValueError(f"simulation.{key} must be Boolean")
    if s["physics_hz"] % s["render_hz"] or s["render_hz"] % c["camera"]["fps"]:
        raise ValueError("Require integer physics/render and render/camera rate ratios")
    dimensions = np.array(c["box"]["dimensions"])
    minimum_dimensions = np.asarray(
        c["box"].get("minimum_dimensions", dimensions), float
    )
    maximum_dimensions = np.asarray(
        c["box"].get("maximum_dimensions", dimensions), float
    )
    if (
        minimum_dimensions.shape != (3,)
        or maximum_dimensions.shape != (3,)
        or np.any(minimum_dimensions <= 0)
        or np.any(maximum_dimensions < minimum_dimensions)
    ):
        raise ValueError("Invalid box dimension bounds")
    footprint = (
        np.array(c["pallet"]["spacing_xy"])
        * (np.array([c["pallet"]["columns"], c["pallet"]["rows"]]) - 1)
        + maximum_dimensions[:2]
    )
    if np.any(maximum_dimensions[:2] > np.array(c["pallet"]["spacing_xy"]) - 0.01):
        raise ValueError("Maximum carton footprint exceeds configured pallet slot spacing")
    if np.any(footprint > c["pallet"]["dimensions"][:2]) or np.any(
        np.array(c["pallet"]["spacing_xy"]) < dimensions[:2] + 0.01
    ):
        raise ValueError("Pallet slots overlap or exceed the pallet")
    jitter = float(c["box"]["spawn_jitter_xy"])
    yaws = vector("box", "spawn_yaw_range", 2)
    if not np.isfinite(jitter) or jitter < 0 or yaws[0] > yaws[1]:
        raise ValueError("Invalid spawn randomization")
    if np.any(
        maximum_dimensions[0] * np.sqrt(2) + 2 * jitter
        >= np.array(c["table"]["dimensions"][:2])
    ):
        raise ValueError("Picking table does not contain the randomized box footprint")
    minimum_count = c["process"].get("random_boxes_min", c["pallet"]["rows"] * c["pallet"]["columns"])
    maximum_count = c["process"].get("random_boxes_max", 2 * c["pallet"]["rows"] * c["pallet"]["columns"])
    slot_count = c["pallet"]["rows"] * c["pallet"]["columns"]
    if (
        type(minimum_count) is not int
        or type(maximum_count) is not int
        or minimum_count < slot_count
        or maximum_count < minimum_count
        or maximum_count > 2 * slot_count
    ):
        raise ValueError("Random carton count must allow one or two cartons per pallet column")
    clearance = max(
        c["table"]["dimensions"][2], c["pallet"]["dimensions"][2] + dimensions[2]
    )
    if (
        c["robot"]["travel_z"]
        < clearance + dimensions[2] + c["robot"]["tool_length"] + 0.05
    ):
        raise ValueError("Travel pose lacks package clearance")
    return c
