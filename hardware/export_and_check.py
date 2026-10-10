#!/usr/bin/env python3
"""Export printable parts with OpenSCAD and check the resulting STL topology.

Requires OpenSCAD and NumPy. Checks are geometry checks, not load testing.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PARTS = ("clamp_body", "moving_pad", "link", "pivot", "camera_head")


def read_ascii_stl(path: Path) -> np.ndarray:
    vertices = re.findall(r"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
                          path.read_text())
    if not vertices or len(vertices) % 3:
        raise ValueError(f"Invalid ASCII STL: {path}")
    return np.array(vertices, dtype=float).reshape(-1, 3, 3)


def mesh_check(triangles: np.ndarray) -> dict:
    vertices, indices = np.unique(np.round(triangles.reshape(-1, 3), 5),
                                  axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    edges = Counter()
    direction = Counter()
    parent = list(range(len(vertices)))

    def find(n: int) -> int:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def join(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((int(i), int(j))))] += 1
            direction[(int(i), int(j))] += 1
            join(int(i), int(j))

    area_vectors = np.cross(triangles[:, 1]-triangles[:, 0],
                            triangles[:, 2]-triangles[:, 0])
    signed_volume = np.einsum("ij,ij->i", triangles[:, 0],
                              np.cross(triangles[:, 1], triangles[:, 2])).sum()/6
    report = {
        "triangles": int(len(faces)),
        "dimensions_mm": (vertices.max(axis=0)-vertices.min(axis=0)).round(3).tolist(),
        "minimum_z_mm": float(vertices[:, 2].min()),
        "volume_cm3": round(float(signed_volume)/1000, 3),
        "watertight_edges": all(n == 2 for n in edges.values()),
        "consistent_edge_winding": all(direction[(i, j)] == direction[(j, i)]
                                        for i, j in edges),
        "connected_components": len({find(i) for i in range(len(vertices))}),
        "degenerate_triangles": int((np.linalg.norm(area_vectors, axis=1) < 1e-8).sum()),
    }
    assert report["watertight_edges"], report
    assert report["consistent_edge_winding"], report
    assert report["connected_components"] == 1, report
    assert report["degenerate_triangles"] == 0, report
    assert report["minimum_z_mm"] >= -1e-5, report
    assert signed_volume > 0, report
    return report


def main() -> None:
    openscad = shutil.which("openscad")
    if not openscad:
        raise SystemExit("OpenSCAD not found. Install OpenSCAD to regenerate the STL files.")
    (ROOT / "stl").mkdir(exist_ok=True)
    reports = {}
    for part in PARTS:
        destination = ROOT / "stl" / f"{part}.stl"
        subprocess.run([openscad, "--export-format", "asciistl", "-o", str(destination),
                        "-D", f'part="{part}"', str(ROOT / "hotpot_camera_mount.scad")],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        reports[part] = mesh_check(read_ascii_stl(destination))
        print(f"{part}: {reports[part]['dimensions_mm']} mm; closed, consistently wound, one solid")
    (ROOT / "geometry_check.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
