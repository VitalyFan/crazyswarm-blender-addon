import os
import csv
import math

import bpy
from bpy_extras.io_utils import ExportHelper
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, FloatProperty

bl_info = {
    "name": "crazyswarm2 animation (.csv)",
    "author": "Fan In Su",
    "version": (0, 0, 5),
    "blender": (5, 1, 1),
    #"api": 36079,
    "location": "File > Export > crazyswarm2 animation (.csv)",
    "description": "Export > crazyswarm2 animation (.csv)",
    "warning": "",
    "category": "Import-Export"
}


class ExportCsv(Operator, ExportHelper):
    bl_idname = "export_animation.folder"
    bl_label = "Export crazyswarm2 animation"
    filename_ext = ''
    use_filter_folder = True

    use_namefilter: bpy.props.BoolProperty(
        name="Use name filter for objects",
        default=False,
    )

    drones_name: bpy.props.StringProperty(
        name="Name identifier",
        description="Name identifier for all drone objects",
        default="cf"
    )

    segment_duration: bpy.props.FloatProperty(
        name="Segment duration (s)",
        description="Length of one polynomial segment in seconds",
        unit='TIME',
        default=0.4,
        min=0.1,
    )

    filepath: StringProperty(
        name="File Path",
        description="File path used for exporting csv files",
        maxlen=1024,
        subtype='DIR_PATH',
        default=""
    )

    def execute(self, context):

        create_folder_if_does_not_exist(self.filepath)
        scene = context.scene
        objects = context.visible_objects

        drone_objects = []
        if self.use_namefilter:
            for drone_obj in objects:
                if self.drones_name.lower() in drone_obj.name.lower():
                    drone_objects.append(drone_obj)
        else:
            drone_objects = objects

        frame_start = scene.frame_start
        frame_end = scene.frame_end
        fps = scene.render.fps / scene.render.fps_base
        dt = 1.0 / fps

        samples = {}
        for drone_obj in drone_objects:
            samples[drone_obj] = {
                "t": [],
                "x": [],
                "y": [],
                "z": [],
                "yaw": [],
            }

        for frame_number in range(frame_start, frame_end + 1):
            scene.frame_set(frame_number)
            t = (frame_number - frame_start) * dt
            for drone_obj in drone_objects:
                x, y, z = drone_obj.matrix_world.to_translation()
                # rot_z = drone_obj.matrix_world.to_euler('XYZ')[2]
                rot_z = 0.0
                samples[drone_obj]["t"].append(t)
                samples[drone_obj]["x"].append(x)
                samples[drone_obj]["y"].append(y)
                samples[drone_obj]["z"].append(z)
                samples[drone_obj]["yaw"].append(rot_z)

        for drone_obj in drone_objects:
            out_path = os.path.join(self.filepath, '{}.csv'.format(drone_obj.name.lower()))
            write_trajectory_csv(
                out_path,
                samples[drone_obj],
                self.segment_duration,
            )
            self.report({'WARNING'}, "Trajectory exported for drone '%s'" % drone_obj.name)
        return {'FINISHED'}


def create_folder_if_does_not_exist(folder_path):
    if os.path.isdir(folder_path):
        return
    os.mkdir(folder_path)


POLY_DEGREE = 7
NUM_COEFFS = POLY_DEGREE + 1
MAX_SEGMENTS = 62

CSV_HEADER = (
    ["duration"]
    + [f"x^{i}" for i in range(NUM_COEFFS)]
    + [f"y^{i}" for i in range(NUM_COEFFS)]
    + [f"z^{i}" for i in range(NUM_COEFFS)]
    + [f"yaw^{i}" for i in range(NUM_COEFFS)]
)


def solve_linear_system(a, b):
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]

    for col in range(n):
        pivot = col
        for row in range(col + 1, n):
            if abs(a[row][col]) > abs(a[pivot][col]):
                pivot = row
        if abs(a[pivot][col]) < 1e-12:
            return [0.0] * n
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]

        inv = 1.0 / a[col][col]
        for k in range(col, n):
            a[col][k] *= inv
        b[col] *= inv

        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if abs(factor) < 1e-12:
                continue
            for k in range(col, n):
                a[row][k] -= factor * a[col][k]
            b[row] -= factor * b[col]

    return b


def fit_poly7_vandermonde(samples_t, samples_v):
    a = []
    for t in samples_t:
        a.append([t ** i for i in range(NUM_COEFFS)])
    return solve_linear_system(a, list(samples_v))


def fit_poly7_least_squares(samples_t, samples_v):
    a = []
    for t in samples_t:
        a.append([t ** i for i in range(NUM_COEFFS)])

    ata = [[0.0 for _ in range(NUM_COEFFS)] for _ in range(NUM_COEFFS)]
    atb = [0.0 for _ in range(NUM_COEFFS)]

    for row, b in zip(a, samples_v):
        for i in range(NUM_COEFFS):
            atb[i] += row[i] * b
            for j in range(NUM_COEFFS):
                ata[i][j] += row[i] * row[j]

    return solve_linear_system(ata, atb)


def poly7_boundary(seg_duration, p0, v0, a0, j0, p1, v1, a1, j1):
    def row(t, der):
        r = []
        for i in range(NUM_COEFFS):
            if i < der:
                r.append(0.0)
                continue
            m = 1.0
            for k in range(der):
                m *= i - k
            r.append(m * (t ** (i - der)))
        return r

    a = []
    b = []

    for der, val in [(0, p0), (1, v0), (2, a0), (3, j0)]:
        a.append(row(0.0, der))
        b.append(val)

    for der, val in [(0, p1), (1, v1), (2, a1), (3, j1)]:
        a.append(row(seg_duration, der))
        b.append(val)

    return solve_linear_system(a, b)


def sample_at(times, values, t, dt):
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    idx = int(t / dt)
    idx = max(0, min(idx, len(times) - 2))
    t0 = times[idx]
    t1 = times[idx + 1]
    v0 = values[idx]
    v1 = values[idx + 1]
    if t1 <= t0:
        return v0
    alpha = (t - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)


def velocity_at(times, values, t, dt):
    if t - dt < times[0]:
        p0 = sample_at(times, values, t, dt)
        p1 = sample_at(times, values, t + dt, dt)
        return (p1 - p0) / dt
    if t + dt > times[-1]:
        p0 = sample_at(times, values, t - dt, dt)
        p1 = sample_at(times, values, t, dt)
        return (p1 - p0) / dt
    p0 = sample_at(times, values, t - dt, dt)
    p1 = sample_at(times, values, t + dt, dt)
    return (p1 - p0) / (2.0 * dt)


def acceleration_at(times, values, t, dt):
    if t - dt < times[0]:
        p0 = sample_at(times, values, t, dt)
        p1 = sample_at(times, values, t + dt, dt)
        p2 = sample_at(times, values, t + 2.0 * dt, dt)
        return (p2 - 2.0 * p1 + p0) / (dt * dt)
    if t + dt > times[-1]:
        p0 = sample_at(times, values, t - 2.0 * dt, dt)
        p1 = sample_at(times, values, t - dt, dt)
        p2 = sample_at(times, values, t, dt)
        return (p2 - 2.0 * p1 + p0) / (dt * dt)
    p0 = sample_at(times, values, t - dt, dt)
    p1 = sample_at(times, values, t, dt)
    p2 = sample_at(times, values, t + dt, dt)
    return (p2 - 2.0 * p1 + p0) / (dt * dt)


def build_segment_boundaries(times, samples, target_segments, dt):
    if target_segments <= 1:
        return [times[0], times[-1]]

    weights = []
    for i in range(len(times) - 1):
        dx = samples["x"][i + 1] - samples["x"][i]
        dy = samples["y"][i + 1] - samples["y"][i]
        dz = samples["z"][i + 1] - samples["z"][i]
        speed = math.sqrt(dx * dx + dy * dy + dz * dz) / dt
        weights.append(1e-3 + speed)

    total_weight = sum(weights)
    if total_weight <= 0.0:
        return [times[0], times[-1]]

    boundaries = [times[0]]
    target_step = total_weight / target_segments
    cumulative = 0.0
    next_target = target_step

    for i, w in enumerate(weights):
        prev = cumulative
        cumulative += w
        while next_target <= cumulative and len(boundaries) < target_segments:
            if w <= 0.0:
                frac = 0.0
            else:
                frac = (next_target - prev) / w
            frac = max(0.0, min(1.0, frac))
            t = times[i] + frac * dt
            if t <= boundaries[-1] + 1e-9:
                t = min(times[i + 1], boundaries[-1] + dt)
            boundaries.append(t)
            next_target += target_step

    boundaries.append(times[-1])
    return boundaries


def scale_coeffs(coeffs, seg_duration):
    if seg_duration <= 0:
        return [0.0 for _ in coeffs]
    scaled = []
    for i, c in enumerate(coeffs):
        scaled.append(c / (seg_duration ** i))
    return scaled


def build_segments(samples, segment_duration):
    times = samples["t"]
    if len(times) < 2:
        return []
    dt = times[1] - times[0]
    total_duration = times[-1] - times[0]
    if total_duration <= 0:
        return []

    target_segments = max(1, int(round(total_duration / segment_duration)))
    target_segments = min(target_segments, MAX_SEGMENTS)
    boundaries = build_segment_boundaries(times, samples, target_segments, dt)
    segments = []

    for seg_idx in range(len(boundaries) - 1):
        start_t = boundaries[seg_idx]
        seg_end = boundaries[seg_idx + 1]
        seg_duration = seg_end - start_t
        if seg_duration <= 0.0:
            continue
        x0 = sample_at(times, samples["x"], start_t, dt)
        y0 = sample_at(times, samples["y"], start_t, dt)
        z0 = sample_at(times, samples["z"], start_t, dt)
        yaw0 = sample_at(times, samples["yaw"], start_t, dt)

        x1 = sample_at(times, samples["x"], seg_end, dt)
        y1 = sample_at(times, samples["y"], seg_end, dt)
        z1 = sample_at(times, samples["z"], seg_end, dt)
        yaw1 = sample_at(times, samples["yaw"], seg_end, dt)

        vx0 = velocity_at(times, samples["x"], start_t, dt)
        vy0 = velocity_at(times, samples["y"], start_t, dt)
        vz0 = velocity_at(times, samples["z"], start_t, dt)
        vyaw0 = velocity_at(times, samples["yaw"], start_t, dt)

        vx1 = velocity_at(times, samples["x"], seg_end, dt)
        vy1 = velocity_at(times, samples["y"], seg_end, dt)
        vz1 = velocity_at(times, samples["z"], seg_end, dt)
        vyaw1 = velocity_at(times, samples["yaw"], seg_end, dt)

        ax0 = acceleration_at(times, samples["x"], start_t, dt)
        ay0 = acceleration_at(times, samples["y"], start_t, dt)
        az0 = acceleration_at(times, samples["z"], start_t, dt)
        ayaw0 = acceleration_at(times, samples["yaw"], start_t, dt)

        ax1 = acceleration_at(times, samples["x"], seg_end, dt)
        ay1 = acceleration_at(times, samples["y"], seg_end, dt)
        az1 = acceleration_at(times, samples["z"], seg_end, dt)
        ayaw1 = acceleration_at(times, samples["yaw"], seg_end, dt)

        x_tau = poly7_boundary(seg_duration, x0, vx0, ax0, 0.0, x1, vx1, ax1, 0.0)
        y_tau = poly7_boundary(seg_duration, y0, vy0, ay0, 0.0, y1, vy1, ay1, 0.0)
        z_tau = poly7_boundary(seg_duration, z0, vz0, az0, 0.0, z1, vz1, az1, 0.0)
        yaw_tau = poly7_boundary(seg_duration, yaw0, vyaw0, ayaw0, 0.0, yaw1, vyaw1, ayaw1, 0.0)

        segments.append({
            "duration": seg_duration,
            "x": x_tau,
            "y": y_tau,
            "z": z_tau,
            "yaw": yaw_tau,
        })

    return segments


def write_trajectory_csv(path, samples, segment_duration):
    segments = build_segments(samples, segment_duration)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for seg in segments:
            row = [seg["duration"]]
            row.extend(seg["x"])
            row.extend(seg["y"])
            row.extend(seg["z"])
            row.extend(seg["yaw"])
            writer.writerow([f"{v:.6f}" for v in row])


def menu_func(self, context):
    self.layout.operator(
        ExportCsv.bl_idname,
        text="crazyswarm2 animation (.csv)"
    )


def register():
    bpy.utils.register_class(ExportCsv)
    bpy.types.TOPBAR_MT_file_export.append(menu_func)


def unregister():
    bpy.utils.unregister_class(ExportCsv)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func)


if __name__ == "__main__":
    register()
