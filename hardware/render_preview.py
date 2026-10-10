#!/usr/bin/env python3
"""Render the CAD meshes as a technical preview; requires NumPy and Matplotlib."""
from __future__ import annotations

from pathlib import Path
import os
import tempfile
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hotpot-cad-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hotpot-cad-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from export_and_check import PARTS, read_ascii_stl

ROOT = Path(__file__).resolve().parent


def rotation(axis: str, degrees: float) -> np.ndarray:
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    return {"x": np.array([[1,0,0],[0,c,-s],[0,s,c]]),
            "y": np.array([[c,0,s],[0,1,0],[-s,0,c]]),
            "z": np.array([[c,-s,0],[s,c,0],[0,0,1]])}[axis]


def transform(mesh, matrix=np.eye(3), offset=(0,0,0)):
    return mesh @ matrix.T + np.asarray(offset)


def add_mesh(ax, mesh, color):
    normal = np.cross(mesh[:, 1]-mesh[:, 0], mesh[:, 2]-mesh[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    light = np.array([-0.35,-0.45,0.82])
    brightness = 0.56 + 0.44*np.clip(normal @ light, 0, 1)
    rgb = np.array(to_rgb(color))
    colors = rgb[None, :] * brightness[:, None]
    ax.add_collection3d(Poly3DCollection(mesh, facecolors=colors, edgecolors="none"))


def box(origin, size):
    x, y, z = origin
    a, b, c = size
    vertices = np.array([[x,y,z],[x+a,y,z],[x+a,y+b,z],[x,y+b,z],
                         [x,y,z+c],[x+a,y,z+c],[x+a,y+b,z+c],[x,y+b,z+c]], dtype=float)
    return vertices[np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],
                               [0,1,5],[0,5,4],[1,2,6],[1,6,5],
                               [2,3,7],[2,7,6],[3,0,4],[3,4,7]])]


def decorate(ax, limits, title):
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.set_box_aspect([b-a for a,b in limits])
    ax.set_axis_off()
    ax.set_title(title, fontsize=15, fontweight="bold", color="#19383d", pad=12)
    ax.view_init(elev=22, azim=-48)


def main():
    meshes = {p: read_ascii_stl(ROOT / "stl" / f"{p}.stl") for p in PARTS}
    colors = {"clamp_body":"#edb751", "moving_pad":"#edb751", "link":"#52a7b0",
              "pivot":"#78bec5", "camera_head":"#b1d9dd"}
    fig = plt.figure(figsize=(14,8), facecolor="#f8faf9")
    left = fig.add_subplot(121, projection="3d", facecolor="#f8faf9")
    right = fig.add_subplot(122, projection="3d", facecolor="#f8faf9")

    # Reconstruct the same pose as part="assembly" from the print-oriented STL files.
    clamp = transform(meshes["clamp_body"]-[0,114,16], rotation("x",-90))
    add_mesh(left, clamp, colors["clamp_body"])
    add_mesh(left, transform(meshes["moving_pad"],rotation("x",180),[30,0,42]), colors["moving_pad"])
    theta = 65
    start = np.array([-9,16,99.])
    step = np.array([85*np.cos(np.deg2rad(theta)),8,85*np.sin(np.deg2rad(theta))])
    for origin in (start, start+step):
        arm = transform(meshes["link"]+[42.5,0,-8], rotation("y",-theta) @ rotation("x",90), origin)
        add_mesh(left, arm, colors["link"])
    endpoint = start + 2*step
    add_mesh(left, transform(meshes["pivot"], offset=endpoint-[20,0,20]), colors["pivot"])
    head = transform(meshes["camera_head"]+[12,0,0], rotation("z",90) @ rotation("x",180), endpoint+[-24,20,18])
    add_mesh(left, head, colors["camera_head"])
    # Camera is only a dimensional placeholder, not an included part.
    add_mesh(left, box(endpoint+[-46,15,18], [44,34,24]), "#263b40")
    add_mesh(left, box([27,-3,-24], [6,6,61]), "#9caaa9")
    add_mesh(left, box([0,-55,42], [105,110,28]), "#c6b6a0")
    decorate(left, [(-45,120),(-65,90),(-35,315)], "Adjustable camera fixture")

    offsets={"clamp_body":[25,0,0], "moving_pad":[112,15,0], "link":[130,75,0],
             "pivot":[175,0,0], "camera_head":[225,103,0]}
    for part in PARTS:
        add_mesh(right, transform(meshes[part], offset=offsets[part]), colors[part])
    decorate(right, [(0,275),(-10,140),(0,90)], "Five printable part types")
    right.view_init(elev=46, azim=-58)
    fig.text(.05,.08,"M8 table clamp  /  M6 locking pivots  /  two short links",fontsize=11,color="#52696b")
    fig.text(.53,.08,"Print flat as supplied. Camera and fasteners purchased separately.",fontsize=10,color="#52696b")
    fig.text(.05,.025,"Prototype CAD · mm · Fit and load testing still required",fontsize=10,color="#687c7e")
    fig.subplots_adjust(left=.02,right=.98,top=.92,bottom=.11,wspace=.02)
    fig.savefig(ROOT / "camera_mount_preview.png",dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(ROOT / "camera_mount_preview.png")


if __name__ == "__main__":
    main()
