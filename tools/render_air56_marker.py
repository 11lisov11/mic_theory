"""Render FreeCAD geometry and solved AIR56 fields to PNG and MP4."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
import meshio
import numpy as np


BACKGROUND = "#f5f7fa"
PALETTE = {"Housing": "#305969", "Stator": "#788ea1", "Rotor": "#587185",
           "Front": "#447684", "Rear": "#447684"}
WIDE = (11, 6)
LIGHT = np.array([-0.3, 0.65, 0.7])
LIGHT /= np.linalg.norm(LIGHT)


def surface(path):
    data = json.loads(path.read_text(encoding="utf8"))
    parts = []
    for row in data:
        triangles = np.asarray(row["vertices"])[np.asarray(row["faces"])]
        parts.append((row["group"], row["part"], triangles, to_rgb(row["color"])))
    return parts


def camera(parts, rotor_angle=0.0, explosion=0.0, cut=False, marker=True):
    angle = np.deg2rad(rotor_angle)
    rotor_spin = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    right = np.array([0.30, -0.28, 0.912])
    right /= np.linalg.norm(right)
    up = np.array([0.15, 0.96, 0.0])
    up -= np.dot(up, right)*right
    up /= np.linalg.norm(up)
    depth = np.cross(right, up)
    pixels, zbuffer, colors = [], [], []
    for group, name, triangles, rgb in parts:
        if not marker and name.startswith("InterlaminarBridge"):
            continue
        if group == "Rotor":
            triangles = triangles @ rotor_spin.T
        if cut and group != "Rotor":
            triangles = triangles[triangles[:, :, 0].mean(axis=1) > 2.]
        if not len(triangles):
            continue
        if explosion:
            shift = {"Front": -65, "Housing": -25, "Stator": 40, "Rotor": 110, "Rear": 173}[group]
            triangles = triangles+np.array([0, 0, explosion*shift])
        v1, v2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
        normal = np.cross(v1, v2)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1)[:, None], 1e-15)
        facing = normal @ depth > 0
        triangles, normal = triangles[facing], normal[facing]
        lambert = 0.55+0.45*np.clip(normal@LIGHT, 0, 1)
        pixels.append(triangles @ np.c_[right, up])
        zbuffer.append((triangles @ depth).mean(axis=1))
        colors.append(np.array(rgb)[None, :]*lambert[:, None])
    order = np.argsort(np.concatenate(zbuffer))
    return np.concatenate(pixels)[order], np.concatenate(colors)[order]


def render_geometry(cad, out, angles, trace_data):
    parts = surface(cad/"cad_scene.json")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=BACKGROUND, layout="constrained")
    views = [(False, 0, True, "AIR56 reconstruction"), (False, 1, True, "Exploded assembly"),
             (True, 0, False, "Section · isolated laminations"), (True, 0, True, "Section · bridge variant")]
    for ax, (cut, exploded, marker, title) in zip(axes.ravel(), views):
        xy, color = camera(parts, explosion=exploded, cut=cut, marker=marker)
        ax.add_collection(PolyCollection(xy, facecolors=color, edgecolors="none", antialiaseds=False))
        ax.autoscale_view()
        ax.set(aspect="equal", title=title)
        ax.axis("off")
    fig.savefig(out/"air56_assembly.png", dpi=190, facecolor=BACKGROUND)
    plt.close(fig)

    fig, (ax, chart) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.65, 1]},
                                    facecolor=BACKGROUND, layout="constrained")
    ax.set(aspect="equal", xlim=(-85, 90), ylim=(-60, 65))
    ax.axis("off")
    xy, color = camera(parts, cut=True)
    collection = PolyCollection(xy, facecolors=color, edgecolors="none", antialiaseds=False)
    ax.add_collection(collection)
    ax.set_title("AIR56 · reconstructed rotor")
    y = (angles[:, 1]-angles[:, 1].mean())*1e6
    chart.plot(angles[:, 0], y, color="#285f7b", lw=1.8)
    chart.set(xlabel="Mechanical angle, deg", ylabel="Centered Re ΔZαα, µΩ",
              title=f"Reduced FEM · {float(trace_data['frequency_hz']):g} Hz")
    chart.set_xlim(0, 360)
    chart.grid(alpha=0.2)
    marker, = chart.plot([0], [y[0]], "o", color="#9c4327", markersize=7)
    title = ax.text(0.03, 0.03, "", transform=ax.transAxes, color="#345065")

    def update(frame):
        angle = 4*frame
        xy, color = camera(parts, rotor_angle=angle, cut=True)
        collection.set_verts(xy)
        collection.set_facecolor(color)
        marker.set_data([angle], [y[angle]])
        title.set_text(f"Mechanical position  {angle:d}°")
        return collection, marker, title

    FuncAnimation(fig, update, frames=90, cache_frame_data=False).save(
        out/"air56_rotor_and_signature.mp4", writer=FFMpegWriter(fps=20), dpi=105)
    plt.close(fig)


def values_from_vtu(path):
    mesh = meshio.read(path)
    get = lambda key: mesh.cell_data_dict[key]["tetra"]
    xyz = mesh.points[mesh.cells_dict["tetra"]].mean(axis=1)*1e3
    mask = (abs(xyz[:, 0]) < 0.45) & (get("material") != 0)
    j = get("J_real_A_m2") + 1j*get("J_imag_A_m2")
    b = get("B_real_T") + 1j*get("B_imag_T")
    heat = get("heat_W_m3")
    return xyz[mask], get("material")[mask], j[mask], b[mask], heat[mask]


def plot_fields(fem, out):
    paths = [fem/"coupon_control_2500hz.vtu", fem/"coupon_marker_2500hz.vtu"]
    data = [values_from_vtu(p) for p in paths]
    fields = [np.column_stack((np.linalg.norm(j, axis=1), np.linalg.norm(b, axis=1)*1e3, heat))
              for _, _, j, b, heat in data]
    vmax = np.maximum(np.vstack(fields).max(axis=0), 1e-16)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.7), layout="constrained")
    titles = ["|J| phasor, A/m²", "|B| phasor, mT", "Cycle-average heat, W/m³"]
    for row, ((xyz, mat, _, _, _), field_values) in enumerate(zip(data, fields)):
        for column, ax in enumerate(axes[row]):
            p = ax.scatter(xyz[:, 1], xyz[:, 2], c=field_values[:, column],
                           s=14, cmap="viridis", vmin=0, vmax=vmax[column])
            if row:
                bridge = mat == 2
                ax.scatter(xyz[bridge, 1], xyz[bridge, 2], facecolors="none", edgecolors="#b94d28",
                           s=17, lw=0.65, label="Contact cells")
            txt = "Isolated sheets" if not row else "Two bridges"
            ax.set(xlabel="Tangential y, mm", ylabel="Stack z, mm",
                   title=txt + " · " + titles[column], ylim=(-0.44, 0.44))
            fig.colorbar(p, ax=ax)
    fig.suptitle("3D FEM coupon · 2.5 kHz · center radial slice")
    fig.savefig(out/"coupon_fields.png", dpi=180, facecolor=BACKGROUND)
    plt.close(fig)
    xyz, mat, j, b, heat = data[1]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), layout="constrained")
    arrows = axes[0].quiver(xyz[:, 1], xyz[:, 2], j.real[:, 1], j.real[:, 2], np.linalg.norm(j.real, axis=1),
                            cmap="viridis", clim=(0, vmax[0]), scale=max(vmax[0]*26, 1e-16), width=0.003)
    field = axes[1].scatter(xyz[:, 1], xyz[:, 2], c=np.linalg.norm(b.real, axis=1)*1e3, s=18,
                            cmap="viridis", vmin=0, vmax=vmax[1])
    q = axes[2].scatter(xyz[:, 1], xyz[:, 2], c=heat, s=18, cmap="magma", vmin=0, vmax=vmax[2])
    for ax, quantity in zip(axes, (arrows, field, q)):
        ax.set(xlabel="Tangential y, mm", ylabel="Stack z, mm", ylim=(-0.44, 0.44))
        fig.colorbar(quantity, ax=ax)
    axes[1].set_title("Instantaneous |B|, mT")
    axes[2].set_title("Time-averaged Joule heat, W/m³")
    top = axes[0].set_title("")
    for ax in axes:
        ax.scatter(xyz[mat == 2, 1], xyz[mat == 2, 2], s=15, edgecolors="#e66b4a", facecolors="none")

    def update(frame):
        phase = 2*np.pi*frame/60
        current = (j*np.exp(1j*phase)).real
        arrows.set_UVC(current[:, 1], current[:, 2], np.linalg.norm(current, axis=1))
        field.set_array(np.linalg.norm((b*np.exp(1j*phase)).real, axis=1)*1e3)
        top.set_text(f"Instantaneous J, A/m² · phase {np.rad2deg(phase):.0f}°")
        return arrows, field, top

    FuncAnimation(fig, update, frames=60, cache_frame_data=False).save(
        out/"coupon_eddy_current.mp4", writer=FFMpegWriter(fps=16), dpi=110)
    plt.close(fig)


def animate_contacts(fem, out):
    mesh = meshio.read(fem/"coupon_marker_2500hz.vtu")
    kind = mesh.cell_data_dict["material"]["tetra"]
    centers = mesh.points[mesh.cells_dict["tetra"]].mean(axis=1)*1e3
    contacts = centers[kind == 2]
    steel_edges = mesh.points[mesh.cells_dict["tetra"][kind == 1]].reshape(-1, 3)*1e3
    xmin, ymin, zmin = steel_edges.min(axis=0)
    xmax, ymax, zmax = steel_edges.max(axis=0)
    # Exact sheet bounds come from the mesh, including the insulation gap.
    positive = steel_edges[:, 2][steel_edges[:, 2] > 0]
    gap = 2*positive.min()
    thickness = zmax-gap/2
    y_positions = [np.mean(contacts[contacts[:, 1] < 0, 1]), np.mean(contacts[contacts[:, 1] > 0, 1])]
    span = (mesh.points[mesh.cells_dict["tetra"][kind == 2]].reshape(-1, 3)*1e3)
    xx = np.unique(span[:, 0])
    width = xx[-1]-xx[0]
    fig = plt.figure(figsize=(10, 6), facecolor=BACKGROUND)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        state = frame//18
        split = max(0., 1-frame/17)*2.3
        ax.bar3d(xmin, ymin, zmin-split, xmax-xmin, ymax-ymin, thickness, color="#607f98")
        ax.bar3d(xmin, ymin, gap/2+split, xmax-xmin, ymax-ymin, thickness, color="#7793a6")
        for y in y_positions[:state]:
            ax.bar3d(xx[0], y-width/2, -gap/2, width, width, gap, color="#e96448", shade=False)
            ax.plot([0, 0], [y, y], [-0.45, 0.45], color="#af422e", linewidth=2, linestyle=":")
        ax.set(xlim=(-3.1, 3.1), ylim=(-6.1, 6.1), zlim=(-2.8, 2.8),
               xlabel="Radial x, mm", ylabel="Tangential y, mm", zlabel="Stack z, mm")
        ax.set_box_aspect((6, 12, 5.6))
        ax.view_init(27, -36)
        title = ("Isolated sheets", "Single bridge", "Two bridges · interlayer loop")[min(state, 2)]
        ax.set_title(title, fontsize=15, pad=12)
        if frame == 17:
            fig.savefig(out/"coupon_geometry.png", dpi=170, facecolor=BACKGROUND)
        return ()

    FuncAnimation(fig, update, frames=54, cache_frame_data=False).save(
        out/"coupon_contact_topology.mp4", writer=FFMpegWriter(fps=16), dpi=105)
    plt.close(fig)

def observer_animation(trace, out):
    fig, axes = plt.subplots(2, 1, figsize=WIDE, layout="constrained")
    index = np.arange(len(trace["truth_rad"]))
    axes[0].plot(index, np.rad2deg(trace["truth_rad"] % np.pi), color="#285f7b")
    axes[0].set_title("Reference modulo 180° · scoring only")
    axes[1].plot(index, np.rad2deg(trace["map_rad"]), color="#a65e29")
    axes[1].set(title="Observer MAP from phasor current · unqualified when the posterior is broad",
                xlabel="Probe index")
    cursors = [ax.axvline(0, color="#74818f", lw=1.5) for ax in axes]
    for ax in axes:
        ax.set(ylabel="Mechanical angle, deg", ylim=(-5, 185), xlim=(0, len(index)-1))
        ax.grid(alpha=0.2)

    def update(frame):
        for cursor in cursors:
            cursor.set_xdata([frame, frame])
        return cursors

    FuncAnimation(fig, update, frames=index[::2], cache_frame_data=False).save(
        out/"observer_reference_and_estimate.mp4", writer=FFMpegWriter(fps=20), dpi=105)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cad", "fem", "observability", "out"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    with np.load(args.observability/"observer_trace.npz") as trace:
        render_geometry(args.cad, args.out, np.loadtxt(args.observability/"angular_map.csv", delimiter=",", skiprows=1),
                        trace)
        plot_fields(args.fem, args.out)
        animate_contacts(args.fem, args.out)
        observer_animation(trace, args.out)


if __name__ == "__main__":
    main()
