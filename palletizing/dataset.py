"""Rendered masks -> audited YOLO instance-segmentation labels."""

import numpy as np


def instance_polygons(segmentation, body_ids, min_pixels=100):
    import cv2

    encoded = np.asarray(segmentation, dtype=np.int64)
    bodies = np.where(encoded >= 0, encoded & ((1 << 24) - 1), -1)
    h, w = bodies.shape
    labels = []
    instances = np.zeros((h, w), np.uint16)
    for instance, body in enumerate(body_ids, 1):
        mask = (bodies == body).astype(np.uint8)
        if int(mask.sum()) < min_pixels:
            raise ValueError("Object has too few visible pixels")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(
            contour, 0.001 * cv2.arcLength(contour, True), True
        ).reshape(-1, 2)
        if len(polygon) < 3:
            raise ValueError("Degenerate visible polygon")
        reconstructed = np.zeros_like(mask)
        cv2.fillPoly(reconstructed, [polygon.astype(np.int32)], 1)
        iou = np.logical_and(mask, reconstructed).sum() / max(
            1, np.logical_or(mask, reconstructed).sum()
        )
        if iou < 0.985:
            raise ValueError(
                "Occlusion splits the instance or creates a hole; polygon would mislabel pixels"
            )
        normal = polygon / np.array([w, h])
        labels.append("0 " + " ".join(f"{x:.7f}" for x in normal.ravel()))
        instances[mask.astype(bool)] = instance
    return labels, instances
