#!/usr/bin/env python3
"""Render verified pallet placements from a simulation report as an SVG."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


PALETTE = ["#167d9a", "#e09f3e", "#6a8f5b", "#8e6cab", "#cf5c55"]


def render(report: dict, config: dict) -> str:
    placements = report.get("placements", [])
    if not placements:
        raise ValueError("Report has no placements")
    requested = int(report.get("requested", len(placements)))
    committed = int(report.get("committed", len(placements)))
    if report.get("state") != "DONE" or committed != requested:
        raise ValueError(
            f"Only complete DONE reports can be rendered as verified results "
            f"(state={report.get('state')}, committed={committed}/{requested})"
        )
    layers = max(int(p["layer"]) for p in placements) + 1

    pallet = config["pallet"]
    center_x, center_y = map(float, pallet["center_xy"])
    pallet_x, pallet_y, pallet_h = map(float, pallet["dimensions"])
    base_z = float(pallet_h)
    x0, x1 = center_x - pallet_x / 2, center_x + pallet_x / 2
    y0, y1 = center_y - pallet_y / 2, center_y + pallet_y / 2

    # Fixed canvas with matched top and front projections.
    width, height = 1200, 660
    top = dict(x=85, y=185, w=490, h=330)
    front = dict(x=690, y=185, w=430, h=330)
    z_max = max(float(p["center"][2]) + float(p["dimensions"][2]) / 2 for p in placements)
    z_max = max(z_max * 1.12, base_z + 0.2)

    def text(x, y, value, size=16, color="#23313b", weight="400", anchor="start"):
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
                f'font-size="{size}" fill="{color}" font-weight="{weight}" '
                f'text-anchor="{anchor}">{html.escape(str(value))}</text>')

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="24" fill="#f5f7f8"/>',
        text(60, 62, "UR10e palletizing simulation", 28, "#14232c", "700"),
        text(60, 94, f"Verified final layout · {committed}/{requested} cartons · seed {report.get('seed', 'n/a')}", 16, "#51626d"),
        '<line x1="60" y1="120" x2="1140" y2="120" stroke="#d6dfe3"/>',
        text(top["x"], 158, "TOP VIEW · pallet footprint", 15, "#51626d", "700"),
        text(front["x"], 158, "FRONT VIEW · stack height", 15, "#51626d", "700"),
    ]
    for layer in range(layers):
        lx = 820 + layer * 95
        parts.append(f'<rect x="{lx}" y="78" width="13" height="13" rx="3" fill="{PALETTE[layer % len(PALETTE)]}"/>')
        parts.append(text(lx + 20, 90, f"Layer {layer + 1}", 12, "#51626d"))

    def tx(x):
        return top["x"] + (x - x0) / pallet_x * top["w"]

    def ty(y):
        return top["y"] + top["h"] - (y - y0) / pallet_y * top["h"]

    # Pallet board in top and side projections.
    parts.append(f'<rect x="{tx(x0):.2f}" y="{ty(y1):.2f}" width="{top["w"]:.2f}" height="{top["h"]:.2f}" rx="4" fill="#dce3e6" stroke="#80919a" stroke-width="2"/>')
    parts.append(text(top["x"] + top["w"] / 2, top["y"] + top["h"] + 28, f"{pallet_x:.2f} m × {pallet_y:.2f} m", 13, "#687982", anchor="middle"))

    def fx(x):
        return front["x"] + (x - x0) / pallet_x * front["w"]

    def fz(z):
        usable_h = front["h"] - 16
        return front["y"] + front["h"] - (z / z_max) * usable_h

    ground_y = fz(base_z)
    parts.append(f'<rect x="{front["x"]}" y="{ground_y:.2f}" width="{front["w"]}" height="{front["y"] + front["h"] - ground_y:.2f}" fill="#dce3e6" stroke="#80919a" stroke-width="2"/>')
    parts.append(f'<line x1="{front["x"]}" y1="{front["y"] + front["h"]:.2f}" x2="{front["x"] + front["w"]}" y2="{front["y"] + front["h"]:.2f}" stroke="#80919a" stroke-width="2"/>')

    # Draw low layers first so upper cartons remain visible.
    for item in sorted(placements, key=lambda p: (int(p["layer"]), float(p["center"][2]), int(p["index"]))):
        index = int(item["index"])
        cx, cy, cz = map(float, item["center"])
        dx, dy, dz = map(float, item["dimensions"])
        layer = int(item["layer"])
        color = PALETTE[layer % len(PALETTE)]
        left, right = cx - dx / 2, cx + dx / 2
        low_y, high_y = cy - dy / 2, cy + dy / 2
        # Top view rectangle.
        rx, ry = tx(left), ty(high_y)
        rw, rh = (right - left) / pallet_x * top["w"], (high_y - low_y) / pallet_y * top["h"]
        parts.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{rw:.2f}" height="{rh:.2f}" rx="5" fill="{color}" fill-opacity="0.88" stroke="#ffffff" stroke-width="2"/>')
        if rw > 38 and rh > 24:
            parts.append(text(rx + rw / 2, ry + rh / 2 + 5, f"B{index:02d}", 12, "#ffffff", "700", "middle"))
        # Front projection along X.
        bx, by = fx(left), fz(cz + dz / 2)
        bw, bh = (right - left) / pallet_x * front["w"], dz / z_max * (front["h"] - 16)
        parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bh:.2f}" rx="3" fill="{color}" fill-opacity="0.82" stroke="#ffffff" stroke-width="1.5"/>')
        if bw > 34 and bh > 20:
            parts.append(text(bx + bw / 2, by + bh / 2 + 4, f"{index:02d}", 11, "#ffffff", "700", "middle"))

    utilization = float(report.get("packing_volume_utilization", 0.0)) * 100
    top_z = max(float(p["center"][2]) + float(p["dimensions"][2]) / 2 for p in placements)
    parts.extend([
        text(60, 590, f"Layers: {layers}", 15, "#23313b", "700"),
        text(240, 590, f"Top height: {top_z:.2f} m", 15, "#23313b", "700"),
        text(495, 590, f"Volume utilization: {utilization:.1f}%", 15, "#23313b", "700"),
        text(60, 625, "Geometry is taken from the saved simulation report; this figure is not a camera image.", 12, "#687982"),
        '</svg>',
    ])
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    config = json.loads(args.config.read_text()) if args.config.suffix == ".json" else None
    if config is None:
        import yaml
        config = yaml.safe_load(args.config.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report, config), encoding="utf-8")
    print(f"Wrote {args.output} from {len(report['placements'])} verified placements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
