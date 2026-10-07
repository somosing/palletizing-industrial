"""Learned RGB masks select depth; geometry still estimates the grasp pose.

No simulator IDs or body poses enter this module. Geometry failure does not fall
back to the unmasked geometric detector.
"""

from pathlib import Path
import time
import numpy as np
from scipy.spatial import QhullError
from .perception import EyeInHandPerception, fit_box
from .backends.bullet import WristCamera


class MaskedCamera:
    def __init__(self, camera, model, config, device="0", confidence=0.5):
        self.source = camera
        self.model = model
        self.config = config
        self.device = device
        self.confidence = confidence
        self.last_prediction_frame = -2
        self.mask = None
        self.masked_depth = None
        self.prediction_error = "No model inference yet"
        self.latencies = []

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_depth(self):
        depth = self.source.get_depth()
        if depth is None:
            return None
        if self.last_prediction_frame == self.source.number:
            return self.masked_depth
        self.last_prediction_frame = self.source.number
        self.mask = np.zeros(depth.shape, bool)
        self.prediction_error = "No suitable package mask"
        start = time.perf_counter()
        result = self.model.predict(
            source=self.source.rgb[:, :, :3][:, :, ::-1].copy(),
            device=self.device,
            conf=self.confidence,
            imgsz=640,
            rect=False,
            retina_masks=True,
            verbose=False,
        )[0]
        candidates = []
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            if masks.shape[1:] != depth.shape:
                raise RuntimeError(
                    "Model masks must match original camera resolution; retina_masks=True is required"
                )
            for mask in masks:
                mask = mask > 0.5
                near, far = self.config["camera"]["clipping_range"]
                rows, cols = np.nonzero(
                    mask & np.isfinite(depth) & (depth > near) & (depth < far)
                )
                if len(rows) < self.config["camera"]["min_points"]:
                    continue
                points = self.source.get_world_points_from_image_coords(
                    np.c_[cols, rows], depth[rows, cols]
                )
                try:
                    fit_box(points, self.config)
                except (ValueError, QhullError) as error:
                    self.prediction_error = "Predicted mask failed geometry: " + str(
                        error
                    )
                    continue
                candidates.append(mask)
        self.latencies.append(1000 * (time.perf_counter() - start))
        if len(candidates) == 1:
            self.mask = candidates[0]
            self.prediction_error = ""
        elif len(candidates) > 1:
            self.prediction_error = "Ambiguous: multiple pickable packages; this cell feeds one cube at a time"
        self.masked_depth = np.where(self.mask, depth, np.nan)
        return self.masked_depth

    def overlay(self):
        image = self.source.rgb[:, :, :3].copy()
        if self.mask is not None:
            image[self.mask] = (
                image[self.mask] * 0.5 + np.array([20, 240, 60]) * 0.5
            ).astype(np.uint8)
        return image


class LearnedEyeInHandPerception(EyeInHandPerception):
    def __init__(self, p, controller, config, weights, device="0", confidence=0.5):
        from ultralytics import YOLO
        import torch

        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(
                "Train first; checkpoint does not exist: " + str(path)
            )
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable. Install a CUDA-enabled PyTorch build or explicitly use --device cpu"
            )
        model = YOLO(str(path), task="segment")
        if model.task != "segment" or list(model.names.values()) != ["package"]:
            raise ValueError(
                "Use the package segmentation checkpoint produced by train_segmentation.py, not a generic COCO model"
            )
        self.config = config
        self.camera = MaskedCamera(
            WristCamera(p, controller, config), model, config, device, confidence
        )
        self.last_frame = -1
        self.samples = []
        self.last_error = "No model frame yet"

    def initialize(self):
        pass

    def detect(self):
        pose = super().detect()
        if pose is None and self.camera.prediction_error:
            self.last_error = self.camera.prediction_error
        return pose
