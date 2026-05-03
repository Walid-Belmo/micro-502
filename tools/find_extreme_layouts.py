import argparse
import csv
import heapq
import math
import random

import numpy as np


CENTER = np.array([4.0, 4.0])
INNER_RADIUS = 1.5
OUTER_RADIUS = 3.5
HEIGHT_BOUNDS = (0.7, 2.0)
WIDTH_BOUNDS = (0.3, 0.5)
YAW_OFFSET_BOUNDS = (-math.pi / 6.0, math.pi / 6.0)
NUM_GATES = 5
NUM_SEGMENTS = NUM_GATES + 1
SEGMENT_ANGLE = math.pi / NUM_SEGMENTS


def clamp01(value):
    return max(0.0, min(1.0, value))


def angular_bounds(index):
    return (
        ((2 * index - 0.5) * SEGMENT_ANGLE) % (2.0 * math.pi),
        ((2 * index + 0.5) * SEGMENT_ANGLE) % (2.0 * math.pi),
    )


def simulate_layout(seed):
    py_rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    gates = []

    for segment in range(NUM_SEGMENTS):
        lo, hi = angular_bounds(segment)
        if segment == 0:
            angle = py_rng.uniform(lo - 2.0 * math.pi, hi)
        else:
            angle = py_rng.uniform(lo, hi)

        radius = py_rng.uniform(INNER_RADIUS, OUTER_RADIUS)
        x = CENTER[0] - radius * math.cos(angle)
        y = CENTER[1] - radius * math.sin(angle)

        if segment > 0:
            height = float(np_rng.uniform(*HEIGHT_BOUNDS))
            width = float(np_rng.uniform(*WIDTH_BOUNDS))
            yaw_offset = float(np_rng.uniform(*YAW_OFFSET_BOUNDS))
            yaw = angle - math.pi / 2.0 + yaw_offset
            gates.append(
                {
                    "gate": segment - 1,
                    "angle": angle,
                    "radius": radius,
                    "x": x,
                    "y": y,
                    "height": height,
                    "width": width,
                    "yaw": yaw,
                    "yaw_offset": yaw_offset,
                }
            )

    return gates


def score_layout(gates):
    radii = np.array([gate["radius"] for gate in gates])
    heights = np.array([gate["height"] for gate in gates])
    widths = np.array([gate["width"] for gate in gates])
    yaw_offsets = np.array([abs(gate["yaw_offset"]) for gate in gates])

    radius_span = float(np.max(radii) - np.min(radii))
    radius_jump = float(np.sum(np.abs(np.diff(radii))))
    height_span = float(np.max(heights) - np.min(heights))
    height_jump = float(np.sum(np.abs(np.diff(heights))))

    parts = {
        "radius_span": clamp01(radius_span / (OUTER_RADIUS - INNER_RADIUS)),
        "radius_jump": clamp01(radius_jump / (4.0 * (OUTER_RADIUS - INNER_RADIUS))),
        "height_span": clamp01(height_span / (HEIGHT_BOUNDS[1] - HEIGHT_BOUNDS[0])),
        "height_jump": clamp01(height_jump / (4.0 * (HEIGHT_BOUNDS[1] - HEIGHT_BOUNDS[0]))),
        "narrow_average": clamp01((WIDTH_BOUNDS[1] - float(np.mean(widths))) / (WIDTH_BOUNDS[1] - WIDTH_BOUNDS[0])),
        "narrow_minimum": clamp01((WIDTH_BOUNDS[1] - float(np.min(widths))) / (WIDTH_BOUNDS[1] - WIDTH_BOUNDS[0])),
        "yaw_average": clamp01(float(np.mean(yaw_offsets)) / YAW_OFFSET_BOUNDS[1]),
        "yaw_maximum": clamp01(float(np.max(yaw_offsets)) / YAW_OFFSET_BOUNDS[1]),
        "outer_gate": clamp01((float(np.max(radii)) - 3.0) / 0.5),
        "inner_gate": clamp01((2.0 - float(np.min(radii))) / 0.5),
    }

    weights = {
        "radius_span": 1.25,
        "radius_jump": 0.80,
        "height_span": 1.10,
        "height_jump": 0.70,
        "narrow_average": 0.90,
        "narrow_minimum": 0.80,
        "yaw_average": 0.65,
        "yaw_maximum": 0.45,
        "outer_gate": 0.70,
        "inner_gate": 0.70,
    }
    score = sum(parts[name] * weights[name] for name in parts) / sum(weights.values())
    return score, parts


def angle_diff(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def score_transition_layout(gates):
    radii = np.array([gate["radius"] for gate in gates])
    heights = np.array([gate["height"] for gate in gates])
    widths = np.array([gate["width"] for gate in gates])
    yaw_offsets = np.array([abs(gate["yaw_offset"]) for gate in gates])
    points = np.array([[gate["x"], gate["y"], gate["height"]] for gate in gates])

    radius_jumps = np.abs(np.diff(radii))
    height_jumps = np.abs(np.diff(heights))
    headings = []
    for i in range(len(points) - 1):
        delta = points[i + 1, :2] - points[i, :2]
        headings.append(math.atan2(delta[1], delta[0]))

    turn_angles = []
    for i in range(len(headings) - 1):
        turn_angles.append(abs(angle_diff(headings[i + 1], headings[i])))

    angular_steps = []
    for i in range(len(gates) - 1):
        angular_steps.append(abs(angle_diff(gates[i + 1]["angle"], gates[i]["angle"])))

    transition_hardness = []
    for i in range(len(gates) - 1):
        radius_part = clamp01(float(radius_jumps[i]) / (OUTER_RADIUS - INNER_RADIUS))
        height_part = clamp01(float(height_jumps[i]) / (HEIGHT_BOUNDS[1] - HEIGHT_BOUNDS[0]))
        narrow_part = clamp01((WIDTH_BOUNDS[1] - min(gates[i]["width"], gates[i + 1]["width"])) / (WIDTH_BOUNDS[1] - WIDTH_BOUNDS[0]))
        yaw_part = clamp01(max(abs(gates[i]["yaw_offset"]), abs(gates[i + 1]["yaw_offset"])) / YAW_OFFSET_BOUNDS[1])
        angular_part = clamp01(abs(float(angular_steps[i]) - SEGMENT_ANGLE) / SEGMENT_ANGLE)
        if i < len(turn_angles):
            turn_part = clamp01(float(turn_angles[i]) / (0.85 * math.pi))
        else:
            turn_part = angular_part
        transition_hardness.append(
            0.27 * radius_part
            + 0.27 * height_part
            + 0.18 * turn_part
            + 0.12 * angular_part
            + 0.10 * narrow_part
            + 0.06 * yaw_part
        )

    transition_hardness = np.array(transition_hardness)
    parts = {
        "worst_transition_floor": float(np.min(transition_hardness)),
        "transition_average": float(np.mean(transition_hardness)),
        "radius_jump_min": clamp01(float(np.min(radius_jumps)) / (OUTER_RADIUS - INNER_RADIUS)),
        "radius_jump_mean": clamp01(float(np.mean(radius_jumps)) / (OUTER_RADIUS - INNER_RADIUS)),
        "height_jump_min": clamp01(float(np.min(height_jumps)) / (HEIGHT_BOUNDS[1] - HEIGHT_BOUNDS[0])),
        "height_jump_mean": clamp01(float(np.mean(height_jumps)) / (HEIGHT_BOUNDS[1] - HEIGHT_BOUNDS[0])),
        "turn_mean": clamp01(float(np.mean(turn_angles)) / (0.85 * math.pi)) if turn_angles else 0.0,
        "narrow_average": clamp01((WIDTH_BOUNDS[1] - float(np.mean(widths))) / (WIDTH_BOUNDS[1] - WIDTH_BOUNDS[0])),
        "yaw_average": clamp01(float(np.mean(yaw_offsets)) / YAW_OFFSET_BOUNDS[1]),
    }

    weights = {
        "worst_transition_floor": 1.55,
        "transition_average": 1.20,
        "radius_jump_min": 0.90,
        "radius_jump_mean": 0.60,
        "height_jump_min": 0.90,
        "height_jump_mean": 0.60,
        "turn_mean": 0.55,
        "narrow_average": 0.50,
        "yaw_average": 0.30,
    }
    score = sum(parts[name] * weights[name] for name in parts) / sum(weights.values())
    return score, parts


def summarize_layout(gates):
    items = []
    for gate in gates:
        items.append(
            "G{gate} r={radius:.2f} h={height:.2f} w={width:.2f} "
            "yaw_offset_deg={yaw_offset:.1f} pos=({x:.2f},{y:.2f},{height:.2f})".format(
                gate=gate["gate"],
                radius=gate["radius"],
                height=gate["height"],
                width=gate["width"],
                yaw_offset=math.degrees(gate["yaw_offset"]),
                x=gate["x"],
                y=gate["y"],
            )
        )
    return " | ".join(items)


def build_row(seed, strategy):
    gates = simulate_layout(seed)
    if strategy == "transitions":
        score, parts = score_transition_layout(gates)
    else:
        score, parts = score_layout(gates)
    radii = [gate["radius"] for gate in gates]
    heights = [gate["height"] for gate in gates]
    widths = [gate["width"] for gate in gates]
    yaw_offsets = [abs(gate["yaw_offset"]) for gate in gates]
    row = {
        "seed": seed,
        "strategy": strategy,
        "score": score,
        "radius_min": min(radii),
        "radius_max": max(radii),
        "radius_span": max(radii) - min(radii),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_span": max(heights) - min(heights),
        "width_min": min(widths),
        "width_mean": float(np.mean(widths)),
        "yaw_offset_max_deg": math.degrees(max(yaw_offsets)),
        "yaw_offset_mean_deg": math.degrees(float(np.mean(yaw_offsets))),
        "layout": summarize_layout(gates),
    }
    for name, value in parts.items():
        row[f"part_{name}"] = value
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Find hard valid Webots assignment layouts by reproducing the simulator randomization."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50000)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--strategy", choices=["global", "transitions"], default="global")
    parser.add_argument("--csv", default="")
    parser.add_argument("--layout-file", default="")
    args = parser.parse_args()

    top_items = []
    for seed in range(args.start, args.end):
        row = build_row(seed, args.strategy)
        key = (float(row["score"]), int(row["seed"]))
        item = (key, row)
        if len(top_items) < args.top:
            heapq.heappush(top_items, item)
        elif key > top_items[0][0]:
            heapq.heapreplace(top_items, item)

    top_rows = [item[1] for item in top_items]
    top_rows.sort(key=lambda row: row["score"], reverse=True)

    fields = list(top_rows[0].keys()) if top_rows else []
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(top_rows)

    if args.layout_file:
        with open(args.layout_file, "w", encoding="utf-8") as handle:
            for row in top_rows:
                handle.write(f"{row['seed']}\n")

    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:02d}. seed={row['seed']} score={row['score']:.3f} "
            f"radius={row['radius_min']:.2f}-{row['radius_max']:.2f} "
            f"height={row['height_min']:.2f}-{row['height_max']:.2f} "
            f"min_width={row['width_min']:.2f} "
            f"max_yaw_offset={row['yaw_offset_max_deg']:.1f}deg"
        )
        print(f"    {row['layout']}")


if __name__ == "__main__":
    main()
