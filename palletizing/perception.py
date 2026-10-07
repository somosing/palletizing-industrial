"""Depth-only fitting of one complete, upright, square-footprint package."""

from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy import ndimage


@dataclass(frozen=True)
class BoxPose:
    position: np.ndarray
    yaw: float  # modulo pi/2 for an unmarked square
    top_z: float
    point_count: int
    dimensions: np.ndarray | None = None


def _pick_zone_component(points, expected_xy, min_points, cell_size):
    """Keep one connected top-face component nearest the commanded pick zone.

    The finite conveyor can leave other cartons in the wrist camera's field of
    view. Fitting one hull to all visible points merges those cartons into a
    fictitious oversized package. Rasterizing the XY cloud at a small metric
    resolution separates cartons with a physical gap while preserving the
    dense top-face samples of each individual carton.
    """
    points = np.asarray(points, dtype=float)
    if not len(points):
        raise ValueError("No package points in the pick-zone region")
    xy = points[:, :2]
    origin = xy.min(axis=0) - cell_size
    cells = np.floor((xy - origin) / cell_size).astype(np.int32)
    shape = tuple((cells.max(axis=0) + 2).tolist())
    occupied = np.zeros(shape, dtype=bool)
    occupied[cells[:, 0], cells[:, 1]] = True
    # Eight-connectivity tolerates sparse pixels along rotated carton edges.
    labels, _ = ndimage.label(occupied, structure=np.ones((3, 3), dtype=np.uint8))
    point_labels = labels[cells[:, 0], cells[:, 1]]
    counts = np.bincount(point_labels, minlength=int(point_labels.max()) + 1)
    candidates = np.flatnonzero(counts >= int(min_points))
    candidates = candidates[candidates != 0]
    if not len(candidates):
        raise ValueError("No complete carton-sized depth cluster in the pick zone")
    target = np.asarray(expected_xy, dtype=float)
    ranked = []
    for label in candidates:
        cluster = points[point_labels == label]
        centroid = np.median(cluster[:, :2], axis=0)
        ranked.append((float(np.linalg.norm(centroid - target)), -len(cluster), cluster))
    return min(ranked, key=lambda item: (item[0], item[1]))[2]


def fit_box(points, config, known_dimensions=None):
    """Estimate an upright carton pose from its top-face depth points.

    Pass ``known_dimensions`` for the original geometric cube baseline. When
    omitted, infer rectangular footprint and height above the known table plane.
    No simulator object transforms or segmentation IDs are consulted.
    """
    c, table = config["camera"], config["table"]
    center = np.asarray(table["center_xy"], float)
    half = np.asarray(table["dimensions"][:2], float) / 2
    table_top = float(table["dimensions"][2])
    p = np.asarray(points, float)
    roi = np.isfinite(p).all(axis=1) & (np.abs(p[:, :2] - center) < half).all(axis=1)
    if known_dimensions is not None:
        dimensions = np.asarray(known_dimensions, float)
        top_z = table_top + dimensions[2]
        top = p[roi & (np.abs(p[:, 2] - top_z) < 0.012)]
    else:
        minimum = np.asarray(config["box"].get("minimum_dimensions", [0.18, 0.18, 0.15]), float)
        maximum = np.asarray(config["box"].get("maximum_dimensions", [0.32, 0.32, 0.30]), float)
        object_points = p[roi & (p[:, 2] > table_top + minimum[2] * 0.65)
                          & (p[:, 2] < table_top + maximum[2] + c["dimension_tolerance"])]
        if len(object_points) < c["min_points"]:
            raise ValueError("Insufficient package depth points above the table")
        object_points = _pick_zone_component(
            object_points,
            expected_xy=center,
            min_points=c["min_points"],
            cell_size=float(c.get("cluster_cell_size_m", 0.008)),
        )
        top_z = float(np.quantile(object_points[:, 2], 0.995))
        top = object_points[object_points[:, 2] > top_z - 0.009]
        dimensions = None
    if len(top) < c["min_points"]:
        raise ValueError("Insufficient top-face points")
    z = float(np.median(top[:, 2]))
    top = top[np.abs(top[:, 2] - z) < 0.003]
    if len(top) < c["min_points"]:
        raise ValueError("Top face is not planar")
    plane = np.linalg.lstsq(
        np.c_[top[:, :2], np.ones(len(top))], top[:, 2], rcond=None
    )[0]
    if np.linalg.norm(plane[:2]) > math.tan(math.radians(2)):
        raise ValueError("Tilted box; only upright packages are accepted")
    hull = top[ConvexHull(top[:, :2]).vertices, :2]
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2))
    best = None
    for angle in angles:
        cs, sn = np.cos(angle), np.sin(angle)
        rotation = np.array([[cs, -sn], [sn, cs]])
        local = hull @ rotation
        low, high = local.min(axis=0), local.max(axis=0)
        extent = high - low
        area = float(np.prod(extent))
        if best is None or area < best[0]:
            best = (area, ((low + high) / 2) @ rotation.T, extent, float(angle))
    _, xy, extent, yaw = best
    if known_dimensions is not None:
        if np.any(np.abs(extent - dimensions[:2]) > c["dimension_tolerance"]):
            raise ValueError(f"Incomplete or incorrect box footprint: {extent}")
        height = float(dimensions[2])
    else:
        height = float(z - table_top)
        minimum = np.asarray(config["box"].get("minimum_dimensions", [0.18, 0.18, 0.15]), float)
        maximum = np.asarray(config["box"].get("maximum_dimensions", [0.32, 0.32, 0.30]), float)
        dimensions = np.r_[extent, height]
        if np.any(dimensions < minimum - c["dimension_tolerance"]) or np.any(
            dimensions > maximum + c["dimension_tolerance"]
        ):
            raise ValueError(f"Detected carton outside configured dimension range: {dimensions}")
    # Canonical yaw near zero avoids unnecessary wrist rotations by 90 degrees.
    canonical_yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    quarter_turns = int(round((yaw - canonical_yaw) / (np.pi / 2)))
    if quarter_turns % 2:
        dimensions = np.asarray(dimensions).copy()
        dimensions[[0, 1]] = dimensions[[1, 0]]
    yaw = canonical_yaw
    return BoxPose(np.r_[xy, table_top + height / 2], yaw, z, len(top), dimensions)


class EyeInHandPerception:
    def __init__(self, wrist_path, config):
        # Lazy import: the caller must construct SimulationApp first.
        from omni.isaac.sensor import Camera

        self.config = config
        c = config["camera"]
        self.camera = Camera(
            prim_path=wrist_path + "/DepthCamera",
            name="wrist_depth",
            frequency=c["fps"],
            resolution=tuple(c["resolution"]),
        )
        self.camera.set_local_pose(
            translation=np.array(c["local_translation"]),
            orientation=np.array(c["local_orientation_ros"]),
            camera_axes="ros",
        )
        self.camera.set_focal_length(c["focal_length"])
        self.camera.set_horizontal_aperture(c["horizontal_aperture"])
        width, height = c["resolution"]
        self.camera.set_vertical_aperture(c["horizontal_aperture"] * height / width)
        self.camera.set_clipping_range(*c["clipping_range"])
        self.last_frame = -1
        self.samples = []
        self.last_error = "No depth frame received"

    def initialize(self):
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()

    def reset(self):
        self.samples.clear()
        self.last_frame = self.camera.get_current_frame().get("rendering_frame", -1)

    def detect(self):
        """Return a stable BoxPose or None. Call only while scan pose is held."""
        frame = self.camera.get_current_frame()
        number = frame.get("rendering_frame", -1)
        if number is None or number < 0 or number == self.last_frame:
            return None
        self.last_frame = number
        depth = self.camera.get_depth()
        c = self.config["camera"]
        width, height = c["resolution"]
        if depth is None or np.shape(depth) != (height, width):
            self.last_error = "Depth annotator has not produced a correctly sized image"
            return None
        near, far = c["clipping_range"]
        rows, cols = np.nonzero(np.isfinite(depth) & (depth > near) & (depth < far))
        points = self.camera.get_world_points_from_image_coords(
            np.c_[cols, rows], np.asarray(depth)[rows, cols]
        )
        try:
            known = (
                self.config["box"]["dimensions"]
                if getattr(self, "known_dimensions", False)
                else None
            )
            pose = fit_box(points, self.config, known)
        except (ValueError, QhullError) as error:
            self.last_error = str(error)
            self.samples.clear()
            return None
        self.samples.append(pose)
        self.samples = self.samples[-c["stable_frames"] :]
        if len(self.samples) < c["stable_frames"]:
            return None
        positions = np.array([p.position for p in self.samples])
        yaws = np.array([p.yaw for p in self.samples])
        # Fourth-angle statistics respect the cube's 90-degree symmetry.
        yaw = float(np.angle(np.mean(np.exp(4j * yaws))) / 4)
        yaw_errors = np.angle(np.exp(4j * (yaws - yaw))) / 4
        if (
            np.max(np.ptp(positions, axis=0)) > 0.005
            or np.max(np.abs(yaw_errors)) > 0.03
        ):
            self.last_error = "Pose not stable across independent depth frames"
            return None
        return BoxPose(
            np.median(positions, axis=0),
            yaw,
            float(np.median([p.top_z for p in self.samples])),
            pose.point_count,
            np.median([p.dimensions for p in self.samples], axis=0),
        )
