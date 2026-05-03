import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

# The available ground truth state measurements can be accessed by calling sensor_data[item].
# The item values are defined in main.py within read_sensors().


class MyAssignment:
    def __init__(self):
        self.center = np.array([4.0, 4.0])
        self.takeoff_pos = np.array([1.0, 4.0, 1.20])

        self.num_gates = 5
        self.num_segments = self.num_gates + 1
        self.segment_angular_size = math.pi / self.num_segments
        self.nominal_radius = 2.55
        self.inner_radius = 1.5
        self.outer_radius = 3.5

        self.camera_fov = 1.5
        self.camera_offset_body = np.array([0.03, 0.0, 0.01])
        self.opening_height = 0.40
        self.assumed_gate_width = 0.40
        self.replay_speed_limit = 1.85
        self.replay_accel_limit = 6.50
        self.replay_waypoint_tolerance = 0.035
        self.replay_min_width_margin = 0.090
        self.replay_min_height_margin = 0.120
        self.replay_min_bounds_margin = 0.050
        self.replay_plan_attempts = (
            ("normal", 0.58, 0.88, 1.00),
            ("shorter_standoffs", 0.46, 0.68, 1.12),
            ("shortest_slowest", 0.34, 0.50, 1.30),
        )

        self.route_lap = 0
        self.gate_idx = 0
        self.phase = "takeoff"
        self.stage = "search"
        self.started_course = False

        self.commit_until = 0.0
        self.commit_target = None
        self.commit_gate_idx = None
        self.pass_stage = "center"
        self.pass_started_at = 0.0

        self.returning_to_start = False
        self.stored_stage = "entry"
        self.stored_curve_key = None
        self.stored_curve_started_at = 0.0
        self.stored_curve_duration = 1.0
        self.stored_curve_start = None
        self.stored_crossing_key = None
        self.stored_crossing_prev_pos = None
        self.stored_crossing_prev_progress = None
        self.stored_crossing_logged = False
        self.stored_replay_plan = None
        self.stored_replay_lap = None
        self.stored_motion_lap_max_speed = 0.0
        self.stored_motion_lap_max_accel = 0.0
        self.stored_motion_lap_max_speed_context = "none"
        self.stored_motion_lap_max_accel_context = "none"
        self.stored_motion_run_max_speed = 0.0
        self.stored_motion_run_max_accel = 0.0
        self.stored_motion_summary_lap = None

        self.gate_estimates = [None] * self.num_gates
        self.gate_pass_directions = [None] * self.num_gates
        self.gate_pass_direction_sources = ["none"] * self.num_gates
        self.gate_pass_direction_confidence = [0.0] * self.num_gates
        self.estimate_samples = [0] * self.num_gates
        self.ray_observations = [[] for _ in range(self.num_gates)]
        self.estimate_history = [[] for _ in range(self.num_gates)]
        self.gate_estimate_quality = [None] * self.num_gates
        self.min_reliable_estimate_samples = 3
        self.min_reliable_observations = 3
        self.min_reliable_baseline = 0.16
        self.max_reliable_ray_residual = 0.22
        self.max_reliable_recent_spread = 0.26
        self.min_pass_direction_confidence = 0.25

        self.search_started_at = 0.0
        self.search_gate_idx = None
        self.search_probe_idx = 0
        self.search_probe_arrived_at = 0.0
        self.last_detection_time = [-1e9] * self.num_gates
        self.last_observation_time = [-1e9] * self.num_gates
        self.last_detection = None

        base_dir = Path(__file__).resolve().parent
        self.debug_enabled = os.environ.get("MICRO502_DEBUG", "0") != "0"
        self.save_images = os.environ.get("MICRO502_SAVE_IMAGES", "0") != "0"
        self.debug_stdout = os.environ.get("MICRO502_DEBUG_STDOUT", "0") != "0"
        self.run_id = os.environ.get("MICRO502_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
        self.layout_id = os.environ.get("MICRO502_RANDOM_SEED", "unknown")
        self.debug_dir = base_dir / "debug"
        self.log_path = self.debug_dir / "assignment_debug.log"
        self.start_wall = time.time()
        self.log_seq = 0
        self.last_log_time = -1e9
        self.last_image_time = -1e9
        self.image_count = 0
        self.last_state_tuple = None
        self.last_throttled_log = {}

        if self.debug_enabled:
            self.debug_dir.mkdir(exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
        self._log(
            "init "
            f"run_id={self.run_id} layout_id={self.layout_id} "
            "course_strategy=first_lap_detect_then_reuse "
            f"debug_enabled={self.debug_enabled} save_images={self.save_images}"
        )

    def compute_command(self, sensor_data, camera_data, dt):
        now = float(sensor_data.get("t", 0.0))
        pos = self._position(sensor_data)
        yaw = float(sensor_data["yaw"])

        detection = self._detect_gate(camera_data)
        raw_detection = detection
        detection_matches_gate = False
        if (
            detection is not None
            and self.gate_idx < self.num_gates
        ):
            if not self.started_course:
                self._log_throttled(
                    f"reject_detection_gate_{self.gate_idx}_course_not_started",
                    now,
                    0.75,
                    f"detection_rejected gate={self.gate_idx} reason=course_not_started det={self._det_summary(raw_detection)}",
                )
                detection = None
            else:
                detection_matches_gate = self._detection_matches_gate(sensor_data, detection, camera_data.shape, self.gate_idx)
                if not detection_matches_gate:
                    if self.route_lap == 0 and self.commit_gate_idx is None and self._detection_is_search_hint(
                        sensor_data,
                        detection,
                        camera_data.shape,
                        self.gate_idx,
                    ):
                        self._log_throttled(
                            f"search_hint_gate_{self.gate_idx}",
                            now,
                            0.75,
                            f"detection_search_hint gate={self.gate_idx} det={self._det_summary(raw_detection)}",
                        )
                    else:
                        self._log_throttled(
                            f"reject_detection_gate_{self.gate_idx}_does_not_match_expected_gate",
                            now,
                            0.75,
                            f"detection_rejected gate={self.gate_idx} reason=does_not_match_expected_gate det={self._det_summary(raw_detection)}",
                        )
                        detection = None

        if (
            detection is not None
            and detection_matches_gate
            and self.gate_idx < self.num_gates
            and self.started_course
            and self.route_lap == 0
            and self.commit_gate_idx is None
        ):
            self.last_detection = detection
            self.last_detection_time[self.gate_idx] = now
            self._update_gate_estimate(self.gate_idx, sensor_data, detection, camera_data.shape)
            self._save_debug_image(camera_data, detection, now)

        image_shape = camera_data.shape if camera_data is not None else (300, 300, 4)
        command = self._compute_course_command(now, pos, yaw, sensor_data, detection, image_shape)
        self._log_status(now, pos, yaw, detection, command)
        return command

    def _compute_course_command(self, now, pos, yaw, sensor_data, detection, image_shape):
        if not self.started_course:
            if pos[2] < 1.05 or now < 2.0:
                self.phase = "takeoff"
                self.stage = "climb"
                self._log_throttled(
                    "takeoff_climb",
                    now,
                    1.0,
                    f"takeoff_climb pos={self._vec(pos)} target={self._vec(self.takeoff_pos)}",
                )
                return self._limited_setpoint(
                    pos,
                    self.takeoff_pos,
                    max_step=0.35,
                    yaw_target=self._expected_pass_yaw(0),
                )
            self.started_course = True
            self._log(f"course_started t={now:.2f} pos={self._vec(pos)} yaw={yaw:.3f}")

        if pos[2] < 0.62:
            self.phase = "takeoff"
            self.stage = "altitude_recovery"
            recovery = np.array([pos[0], pos[1], 1.15])
            self._log_throttled(
                "altitude_recovery",
                now,
                0.75,
                f"altitude_recovery pos={self._vec(pos)} target={self._vec(recovery)} yaw={yaw:.3f}",
            )
            return self._limited_setpoint(
                pos,
                recovery,
                max_step=0.12,
                yaw_target=yaw,
            )

        if self.route_lap == 0:
            return self._first_lap_command(now, pos, yaw, sensor_data, detection, image_shape)

        if self.route_lap <= 2:
            return self._stored_lap_command(now, pos, yaw, sensor_data, detection, image_shape)

        self.phase = "done"
        self.stage = "hold"
        return self._limited_setpoint(pos, self.takeoff_pos, max_step=0.35, yaw_target=yaw)

    def _first_lap_command(self, now, pos, yaw, sensor_data, detection, image_shape):
        self.phase = "vision_lap"

        if self.gate_idx >= self.num_gates:
            return self._return_to_start_or_next_lap(now, pos, lap_after_return=1)

        if self.commit_gate_idx == self.gate_idx:
            return self._gate_pass_command(now, pos, first_lap=True, detection=detection)

        if detection is None:
            self.stage = "search"
            return self._search_command(now, pos, self.gate_idx)

        self.stage = "vision_servo"
        if self._ready_to_commit(pos, detection):
            if self._start_first_lap_commit(now, pos, yaw, sensor_data, detection, image_shape):
                return self._gate_pass_command(now, pos, first_lap=True, detection=detection)

        return self._vision_servo_command(pos, yaw, detection)

    def _stored_lap_command(self, now, pos, yaw, sensor_data, detection, image_shape):
        self.phase = f"stored_lap_{self.route_lap}"

        if any(gate is None for gate in self.gate_estimates):
            self.stage = "fallback_search"
            return self._search_command(now, pos, self.gate_idx)

        if self.stored_replay_plan is None or self.stored_replay_lap != self.route_lap:
            self.stored_replay_plan = self._build_stored_replay_plan(now, pos)
            self.stored_replay_lap = self.route_lap
            self.gate_idx = 0
            self.stored_stage = "replay"
            self.stored_crossing_key = None
            self._reset_stored_motion_lap_metrics()
            self.stored_motion_summary_lap = None

        plan = self.stored_replay_plan
        elapsed = max(0.0, now - plan["start_time"])
        target, velocity, acceleration = self._sample_stored_replay(plan, elapsed)
        speed = float(np.linalg.norm(velocity))
        accel = float(np.linalg.norm(acceleration))
        total_time = max(float(plan["times"][-1]), 1e-6)
        self._record_stored_motion_metrics(speed, accel, ("replay", self.route_lap), min(elapsed / total_time, 1.0))
        self._update_stored_replay_progress(now, pos, plan, elapsed)

        if elapsed >= total_time:
            if self.route_lap > 0:
                self._log_stored_motion_summary_once(self.route_lap, "replay_time_complete")
            return self._return_to_start_or_next_lap(now, pos, lap_after_return=self.route_lap + 1)

        self.stage = "stored_replay"
        yaw_target = math.atan2(velocity[1], velocity[0]) if np.linalg.norm(velocity[:2]) > 0.03 else yaw
        return [float(target[0]), float(target[1]), float(target[2]), float(self._wrap_angle(yaw_target))]

    def _return_to_start_or_next_lap(self, now, pos, lap_after_return):
        self.phase = "return_to_start"
        self.stage = "return"
        segment = self._segment_from_position(pos[:2])
        if segment == 0 and np.linalg.norm(pos[:2] - self.takeoff_pos[:2]) < 0.75:
            self._log(f"own_lap_complete old_lap={self.route_lap} next_lap={lap_after_return}")
            if self.route_lap > 0:
                self._log_stored_motion_summary_once(self.route_lap, "lap_complete")
            self.route_lap = lap_after_return
            self.gate_idx = 0
            self.stage = "search"
            self.stored_stage = "entry"
            self.stored_curve_key = None
            self.stored_crossing_key = None
            self.stored_replay_plan = None
            self.stored_replay_lap = None
            self._reset_stored_motion_lap_metrics()
            self.stored_motion_summary_lap = None
            self.pass_stage = "center"
            if self.route_lap > 2:
                return self._limited_setpoint(pos, self.takeoff_pos, max_step=0.35, yaw_target=0.0)
            return self._search_command(now, pos, self.gate_idx)

        yaw_target = self._yaw_to_point(pos, self.takeoff_pos)
        return self._limited_setpoint(pos, self.takeoff_pos, max_step=0.75, yaw_target=yaw_target)

    def _search_command(self, now, pos, gate_idx):
        if gate_idx >= self.num_gates:
            return self._limited_setpoint(pos, self.takeoff_pos, max_step=0.55, yaw_target=0.0)

        if self.search_gate_idx != gate_idx:
            self.search_gate_idx = gate_idx
            self.search_started_at = now
            self.search_probe_idx = 0
            self.search_probe_arrived_at = 0.0
            self._log(f"search_start gate={gate_idx} expected={self._vec(self._expected_gate_position(gate_idx))}")

        estimate = self.gate_estimates[gate_idx]
        recent_estimate = estimate is not None and now - self.last_detection_time[gate_idx] < 1.0
        if recent_estimate:
            standoff = estimate - 0.55 * self._pass_direction_for_gate(gate_idx, estimate, pos)
            standoff[2] = np.clip(estimate[2], 0.75, 2.05)
            look_at = estimate
            self._log_throttled(
                f"search_using_recent_estimate_{gate_idx}",
                now,
                0.75,
                f"search_using_recent_estimate gate={gate_idx} estimate={self._vec(estimate)} standoff={self._vec(standoff)}",
            )
        else:
            look_at, standoff, probe_label = self._search_probe_target(gate_idx, self.search_probe_idx)
            self._log_throttled(
                f"search_probe_{gate_idx}",
                now,
                1.0,
                f"search_probe gate={gate_idx} probe={self.search_probe_idx} {probe_label} "
                f"look_at={self._vec(look_at)} standoff={self._vec(standoff)}",
            )

        pass_yaw = self._yaw_to_point(pos, look_at)

        dist_xy = np.linalg.norm(pos[:2] - standoff[:2])
        dist_z = abs(pos[2] - standoff[2])
        if dist_xy > 0.32 or dist_z > 0.22:
            if not recent_estimate:
                self.search_probe_arrived_at = 0.0
            self._log_throttled(
                f"search_move_to_standoff_{gate_idx}",
                now,
                1.0,
                f"search_move_to_standoff gate={gate_idx} dist_xy={dist_xy:.2f} dist_z={dist_z:.2f} "
                f"target={self._vec(standoff)} yaw={pass_yaw:.3f}",
            )
            return self._limited_setpoint(pos, standoff, max_step=0.55, yaw_target=pass_yaw)

        if not recent_estimate:
            if self.search_probe_arrived_at <= 0.0:
                self.search_probe_arrived_at = now
            elif now - self.search_probe_arrived_at > 2.0:
                self.search_probe_idx = (self.search_probe_idx + 1) % len(self._search_probe_specs())
                self.search_probe_arrived_at = 0.0
                look_at, standoff, probe_label = self._search_probe_target(gate_idx, self.search_probe_idx)
                pass_yaw = self._yaw_to_point(pos, look_at)
                self._log(
                    f"search_next_probe gate={gate_idx} probe={self.search_probe_idx} {probe_label} "
                    f"look_at={self._vec(look_at)} standoff={self._vec(standoff)}"
                )
                return self._limited_setpoint(pos, standoff, max_step=0.55, yaw_target=pass_yaw)

        scan = 0.28 * math.sin(1.10 * (now - self.search_started_at))
        yaw_target = self._wrap_angle(pass_yaw + scan)
        self._log_throttled(
            f"search_scan_{gate_idx}",
            now,
            1.0,
            f"search_scan gate={gate_idx} standoff={self._vec(standoff)} look_at={self._vec(look_at)} "
            f"yaw={yaw_target:.3f} scan={scan:.3f}",
        )
        return self._limited_setpoint(pos, standoff, max_step=0.18, yaw_target=yaw_target)

    def _search_probe_specs(self):
        edge_offset = 0.42 * self.segment_angular_size
        return [
            (self.nominal_radius, 0.0, 1.35),
            (self.outer_radius - 0.15, -edge_offset, 1.65),
            (self.outer_radius - 0.15, 0.0, 1.75),
            (self.outer_radius - 0.15, edge_offset, 1.65),
            (self.inner_radius + 0.20, -edge_offset, 0.95),
            (self.inner_radius + 0.20, 0.0, 1.05),
            (self.inner_radius + 0.20, edge_offset, 0.95),
            (self.nominal_radius, -edge_offset, 1.85),
            (self.nominal_radius, edge_offset, 0.85),
        ]

    def _search_probe_target(self, gate_idx, probe_idx):
        specs = self._search_probe_specs()
        radius, angle_offset, height = specs[probe_idx % len(specs)]
        angle = self._expected_angle(gate_idx) + angle_offset
        gate = self._gate_position_from_angle_radius(angle, radius, height)
        pass_dir = self._geometric_pass_direction_for_gate(gate_idx, gate)
        standoff = gate - 0.78 * pass_dir
        standoff[0] = np.clip(standoff[0], 0.35, 7.65)
        standoff[1] = np.clip(standoff[1], 0.35, 7.65)
        standoff[2] = np.clip(height, 0.78, 1.95)
        label = f"radius={radius:.2f} angle_offset={angle_offset:.2f} height={height:.2f}"
        return gate, standoff, label

    def _vision_servo_command(self, pos, yaw, detection):
        err_x = detection["err_x"]
        err_y = detection["err_y"]
        area = detection["area"]

        gate = self.gate_estimates[self.gate_idx] if self.gate_idx < self.num_gates else None
        if gate is not None:
            pass_dir = self._pass_direction_for_gate(self.gate_idx, gate, pos)
            pass_yaw = math.atan2(pass_dir[1], pass_dir[0])
            right = np.array([math.sin(pass_yaw), -math.cos(pass_yaw)])
            entry = gate - 0.70 * pass_dir
            dist_entry = np.linalg.norm(pos[:2] - entry[:2])

            if dist_entry < 0.30:
                center_target = pos.copy()
                center_target[:2] += np.clip(0.22 * err_x, -0.12, 0.12) * right
                center_target[:2] += 0.04 * pass_dir[:2]
                center_target[2] = np.clip(pos[2] - np.clip(0.16 * err_y, -0.05, 0.05), 0.70, 2.05)
                yaw_target = self._yaw_to_point(pos, gate)
                self._log_throttled(
                    f"vision_servo_center_{self.gate_idx}",
                    float("inf"),
                    0.0,
                    f"vision_servo_center gate={self.gate_idx} det={self._det_summary(detection)} target={self._vec(center_target)}",
                )
                return self._limited_setpoint(pos, center_target, max_step=0.16, yaw_target=yaw_target)

            entry[:2] += np.clip(0.12 * err_x, -0.07, 0.07) * right
            entry[2] = np.clip(gate[2] - 0.16 * err_y, 0.70, 2.05)
            yaw_target = self._yaw_to_point(pos, gate)
            max_step = 0.34 if dist_entry > 0.45 else 0.18
            self._log_throttled(
                f"vision_servo_entry_{self.gate_idx}",
                float("inf"),
                0.0,
                f"vision_servo_entry gate={self.gate_idx} dist_entry={dist_entry:.2f} det={self._det_summary(detection)} target={self._vec(entry)}",
            )
            return self._limited_setpoint(pos, entry, max_step=max_step, yaw_target=yaw_target)

        yaw_cmd = self._wrap_angle(yaw - 0.35 * err_x)
        forward = np.array([math.cos(yaw_cmd), math.sin(yaw_cmd)])
        right = np.array([math.sin(yaw_cmd), -math.cos(yaw_cmd)])

        center_error = max(abs(err_x), abs(err_y))
        if center_error > 0.35:
            forward_step = 0.03
        elif area < 700:
            forward_step = 0.11
        elif area < 1700:
            forward_step = 0.08
        else:
            forward_step = 0.05

        lateral_step = np.clip(0.12 * err_x, -0.08, 0.08)
        target_xy = pos[:2] + forward_step * forward + lateral_step * right
        z_target = np.clip(pos[2] - np.clip(0.18 * err_y, -0.06, 0.06), 0.70, 2.05)
        target = np.array([target_xy[0], target_xy[1], z_target])
        self._log_throttled(
            f"vision_servo_no_estimate_{self.gate_idx}",
            float("inf"),
            0.0,
            f"vision_servo_no_estimate gate={self.gate_idx} det={self._det_summary(detection)} target={self._vec(target)} yaw={yaw_cmd:.3f}",
        )
        return [float(target[0]), float(target[1]), float(target[2]), float(yaw_cmd)]

    def _ready_to_commit(self, pos, detection):
        if self.gate_idx >= self.num_gates or self.gate_estimates[self.gate_idx] is None:
            self._log_throttled(
                f"commit_not_ready_no_estimate_{self.gate_idx}",
                float("inf"),
                0.75,
                f"commit_not_ready gate={self.gate_idx} reason=no_estimate det={self._det_summary(detection)}",
            )
            return False
        if not self._is_full_gate_detection(detection):
            self._log_throttled(
                f"commit_not_ready_partial_detection_{self.gate_idx}",
                float("inf"),
                0.75,
                f"commit_not_ready gate={self.gate_idx} reason=partial_detection det={self._det_summary(detection)}",
            )
            return False

        gate = self.gate_estimates[self.gate_idx]
        quality_ok, quality_reason, quality = self._estimate_quality_ready_for_commit(self.gate_idx)
        if not quality_ok:
            self._log_throttled(
                f"commit_not_ready_estimate_quality_{self.gate_idx}",
                float("inf"),
                0.75,
                f"commit_not_ready gate={self.gate_idx} reason=estimate_quality:{quality_reason} "
                f"{self._quality_summary(quality)} det={self._det_summary(detection)}",
            )
            return False

        direction_source = self.gate_pass_direction_sources[self.gate_idx]
        direction_confidence = self.gate_pass_direction_confidence[self.gate_idx]
        if direction_source != "vision_pose" or direction_confidence < self.min_pass_direction_confidence:
            self._log_throttled(
                f"commit_not_ready_direction_quality_{self.gate_idx}",
                float("inf"),
                0.75,
                f"commit_not_ready gate={self.gate_idx} reason=direction_quality "
                f"source={direction_source} confidence={direction_confidence:.2f} "
                f"required={self.min_pass_direction_confidence:.2f} det={self._det_summary(detection)}",
            )
            return False

        pass_dir = self._pass_direction_for_gate(self.gate_idx, gate, pos)
        progress = float(np.dot(pos[:2] - gate[:2], pass_dir[:2]))
        if progress > 0.20:
            self._log_throttled(
                f"commit_not_ready_already_past_{self.gate_idx}",
                float("inf"),
                0.75,
                f"commit_not_ready gate={self.gate_idx} reason=already_past progress={progress:.2f} gate={self._vec(gate)} det={self._det_summary(detection)}",
            )
            return False

        right = np.array([-pass_dir[1], pass_dir[0]])
        lateral_error = abs(float(np.dot(pos[:2] - gate[:2], right[:2])))
        before_gate = -progress
        near_center_line = np.linalg.norm(pos[:2] - gate[:2]) < 0.38 and lateral_error < 0.28
        lined_up_for_gate_plane = (0.20 <= before_gate <= 0.95 and lateral_error < 0.34) or near_center_line
        if not lined_up_for_gate_plane:
            self._log_throttled(
                f"commit_not_ready_not_aligned_{self.gate_idx}",
                float("inf"),
                0.75,
                f"commit_not_ready gate={self.gate_idx} reason=not_aligned "
                f"before_gate={before_gate:.2f} lateral={lateral_error:.2f} "
                f"gate={self._vec(gate)} dir={self._vec(pass_dir)} det={self._det_summary(detection)}",
            )
            return False

        err_x = abs(detection["err_x"])
        err_y = abs(detection["err_y"])
        area = detection["area"]
        _, _, bw, bh = detection["bbox"]
        ready = err_x < 0.18 and err_y < 0.20 and area > 850 and bh > 34 and bw > 24
        if ready:
            self._log(
                f"commit_ready gate={self.gate_idx} progress={progress:.2f} "
                f"gate={self._vec(gate)} det={self._det_summary(detection)}"
            )
        else:
            self._log_throttled(
                f"commit_not_ready_thresholds_{self.gate_idx}",
                float("inf"),
                0.75,
                f"commit_not_ready gate={self.gate_idx} reason=thresholds progress={progress:.2f} "
                f"err=({err_x:.2f},{err_y:.2f}) area={area:.0f} bbox=({bw},{bh})",
            )
        return ready

    def _start_first_lap_commit(self, now, pos, yaw, sensor_data, detection, image_shape):
        gate = self.gate_estimates[self.gate_idx]
        visual_gate, visual_reason = self._current_visual_gate_estimate(
            self.gate_idx,
            sensor_data,
            detection,
            image_shape,
            require_full=True,
        )
        if visual_gate is not None:
            if gate is None:
                gate = visual_gate
                self._log(f"commit_visual_estimate gate={self.gate_idx} gate={self._vec(gate)} reason={visual_reason}")
            else:
                shift = np.linalg.norm(gate[:2] - visual_gate[:2])
                if shift > 0.18:
                    self._log(
                        f"commit_recenter_from_vision gate={self.gate_idx} "
                        f"old={self._vec(gate)} visual={self._vec(visual_gate)} shift={shift:.2f}"
                    )
                gate = visual_gate
        else:
            self._log(
                f"commit_no_visual_recenter gate={self.gate_idx} "
                f"reason={visual_reason} current={self._vec(gate)}"
            )
        if gate is None:
            gate = self._monocular_gate_estimate(pos, yaw, detection, (300, 300, 4))
            gate = self._sanitize_gate_estimate(self.gate_idx, gate)

        if gate is not None:
            self._update_pass_direction_from_detection(self.gate_idx, sensor_data, detection, image_shape, gate)
            pass_dir = self._learn_pass_direction(self.gate_idx, gate, pos)
        else:
            self._log(f"commit_blocked_no_valid_estimate gate={self.gate_idx}")
            self.stage = "vision_servo"
            return False

        self.gate_estimates[self.gate_idx] = gate.copy()
        self.estimate_samples[self.gate_idx] = max(self.estimate_samples[self.gate_idx], 1)

        _, center, exit_point, _, _ = self._gate_waypoints(self.gate_idx, gate)

        self.stage = "pass"
        self.pass_stage = "center"
        self.commit_gate_idx = self.gate_idx
        self.commit_until = now + 8.0
        self.pass_started_at = now
        self.commit_target = exit_point
        self._log(
            f"pass_start lap=0 gate={self.gate_idx} "
            f"gate=({gate[0]:.2f},{gate[1]:.2f},{gate[2]:.2f}) "
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
            f"exit=({exit_point[0]:.2f},{exit_point[1]:.2f},{exit_point[2]:.2f}) "
            f"dir=({pass_dir[0]:.2f},{pass_dir[1]:.2f}) "
            f"area={detection['area']:.0f}"
        )
        return True

    def _gate_pass_command(self, now, pos, first_lap, detection=None):
        gate_idx = self.gate_idx
        if gate_idx >= self.num_gates:
            return self._return_to_start_or_next_lap(now, pos, lap_after_return=1)

        gate = self.gate_estimates[gate_idx]
        if gate is None:
            self.stage = "search"
            return self._search_command(now, pos, gate_idx)

        entry, center, exit_point, pass_dir, pass_yaw = self._gate_waypoints(gate_idx, gate)
        if self.pass_stage == "entry":
            self.stage = "pass_entry"
            if np.linalg.norm(pos - entry) < 0.22:
                self._log(f"pass_entry_reached lap=0 gate={gate_idx} pos={self._vec(pos)} entry={self._vec(entry)}")
                self.pass_stage = "center"
            else:
                return self._limited_setpoint(pos, entry, max_step=0.32, yaw_target=pass_yaw)

        if self.pass_stage == "center":
            self.stage = "pass_center"
            center_error = np.linalg.norm(pos - center)
            visually_centered = (
                detection is not None
                and now - self.pass_started_at > 0.35
                and abs(detection["err_x"]) < 0.18
                and abs(detection["err_y"]) < 0.22
                and detection["area"] > 4200
            )
            timed_out = now > self.commit_until - 1.0 and center_error < 0.28
            if center_error < 0.21 or visually_centered or timed_out:
                self._log(f"pass_center_reached lap=0 gate={gate_idx} err={center_error:.2f}")
                self.pass_stage = "exit"
            else:
                return self._limited_setpoint(pos, center, max_step=0.22, yaw_target=pass_yaw)

        self.stage = "pass_exit"
        progress = float(np.dot(pos[:2] - gate[:2], pass_dir[:2]))
        exit_error = np.linalg.norm(pos - exit_point)
        segment = self._segment_from_position(pos[:2])
        crossed_expected_segment = segment == gate_idx + 1
        done_reason = None
        if progress > 0.68:
            done_reason = "clear_progress"
        elif crossed_expected_segment and progress > 0.42:
            done_reason = "next_segment"
        elif exit_error < 0.28 and progress > 0.46:
            done_reason = "near_exit"

        if done_reason is not None:
            self._log(
                f"pass_done lap=0 gate={gate_idx} reason={done_reason} "
                f"progress={progress:.2f} exit_error={exit_error:.2f} segment={segment}"
            )
            self.gate_idx += 1
            self.stage = "search"
            self.pass_stage = "center"
            self.commit_gate_idx = None
            self.commit_target = None
            return self._search_command(now, pos, self.gate_idx)

        if now > self.commit_until and progress < -0.15:
            self._log(
                f"pass_retry_same_gate lap=0 gate={gate_idx} "
                f"progress={progress:.2f} exit_err={exit_error:.2f}"
            )
            self.stage = "vision_servo"
            self.pass_stage = "center"
            self.commit_gate_idx = None
            self.commit_target = None
            self.commit_until = 0.0
            return self._vision_servo_command(pos, pass_yaw, detection) if detection is not None else self._search_command(now, pos, gate_idx)

        return self._limited_setpoint(pos, exit_point, max_step=0.34, yaw_target=pass_yaw)

    def _update_gate_estimate(self, gate_idx, sensor_data, detection, image_shape):
        if not self._is_usable_detection(detection):
            self._log_throttled(
                f"estimate_skip_unusable_detection_{gate_idx}",
                float(sensor_data.get("t", 0.0)),
                0.75,
                f"estimate_skip gate={gate_idx} reason=unusable_detection det={self._det_summary(detection)}",
            )
            return

        now = float(sensor_data.get("t", 0.0))
        if now - self.last_observation_time[gate_idx] < 0.12:
            self._log_throttled(
                f"estimate_skip_too_soon_{gate_idx}",
                now,
                0.75,
                f"estimate_skip gate={gate_idx} reason=too_soon dt={now - self.last_observation_time[gate_idx]:.3f}",
            )
            return
        self.last_observation_time[gate_idx] = now

        camera_pos, ray_dir = self._camera_ray(sensor_data, detection, image_shape)
        observations = self.ray_observations[gate_idx]
        if not observations or np.linalg.norm(camera_pos - observations[-1][0]) > 0.08:
            observations.append((camera_pos, ray_dir))
            if len(observations) > 12:
                observations.pop(0)
            self._log(
                f"observation_added gate={gate_idx} count={len(observations)} "
                f"camera={self._vec(camera_pos)} ray={self._vec(ray_dir)} det={self._det_summary(detection)}"
            )

        mono_raw = self._monocular_gate_estimate_from_ray(camera_pos, ray_dir, detection)
        mono, mono_reason = self._sanitize_gate_estimate_with_reason(gate_idx, mono_raw)
        tri_raw = self._triangulate_rays(observations)
        tri, tri_reason = self._sanitize_gate_estimate_with_reason(gate_idx, tri_raw)
        old = self.gate_estimates[gate_idx]
        candidate = mono
        source = "monocular"
        if mono is not None and tri is not None:
            disagreement = np.linalg.norm(mono[:2] - tri[:2])
            if disagreement <= 0.24:
                candidate = 0.55 * mono + 0.45 * tri
                source = "mono_tri_agree"
            elif len(observations) >= 3 and (
                old is None or np.linalg.norm(tri[:2] - old[:2]) <= np.linalg.norm(mono[:2] - old[:2]) + 0.08
            ):
                candidate = tri
                source = "triangulated_disagreement"
                self._log_throttled(
                    f"estimate_prefer_triangulation_{gate_idx}",
                    now,
                    0.5,
                    f"estimate_prefer_triangulation gate={gate_idx} disagreement={disagreement:.2f} "
                    f"mono={self._vec(mono)} tri={self._vec(tri)}",
                )
            else:
                self._log_throttled(
                    f"estimate_ignore_triangulation_{gate_idx}",
                    now,
                    0.5,
                    f"estimate_ignore_triangulation gate={gate_idx} disagreement={disagreement:.2f} "
                    f"mono={self._vec(mono)} tri={self._vec(tri)}",
                )
        elif tri is not None:
            candidate = tri
            source = "triangulated_fallback"
        if candidate is None:
            self._log_throttled(
                f"estimate_reject_candidate_{gate_idx}",
                now,
                0.5,
                f"estimate_reject gate={gate_idx} mono_raw={self._vec(mono_raw)} mono_reason={mono_reason} "
                f"tri_raw={self._vec(tri_raw)} tri_reason={tri_reason} observations={len(observations)}",
            )
            return

        if old is None:
            estimate = candidate
        elif np.linalg.norm(old[:2] - candidate[:2]) > 0.80:
            self._log(
                f"estimate_reject gate={gate_idx} reason=jump_too_large "
                f"old={self._vec(old)} candidate={self._vec(candidate)} jump={np.linalg.norm(old[:2] - candidate[:2]):.2f}"
            )
            return
        else:
            estimate = 0.72 * old + 0.28 * candidate
        estimate[2] = np.clip(estimate[2], 0.70, 2.05)
        samples_after = self.estimate_samples[gate_idx] + 1
        quality = self._evaluate_gate_estimate_quality(
            gate_idx,
            estimate,
            source,
            detection,
            observations,
            samples_after,
        )
        self.gate_estimates[gate_idx] = estimate
        self.estimate_samples[gate_idx] = samples_after
        self.gate_estimate_quality[gate_idx] = quality
        history = self.estimate_history[gate_idx]
        history.append(estimate.copy())
        if len(history) > 8:
            history.pop(0)
        self._update_pass_direction_from_detection(gate_idx, sensor_data, detection, image_shape, estimate)
        self._log(
            f"estimate_accept gate={gate_idx} source={source} samples={self.estimate_samples[gate_idx]} "
            f"estimate={self._vec(estimate)} candidate={self._vec(candidate)} mono_reason={mono_reason} tri_reason={tri_reason} "
            f"quality={quality['status']} reason={quality['reason']} {self._quality_summary(quality)}"
        )

    def _current_visual_gate_estimate(self, gate_idx, sensor_data, detection, image_shape, require_full):
        if detection is None:
            return None, "no_detection"
        if require_full and not self._is_full_gate_detection(detection):
            return None, "not_full_detection"
        if not require_full and not self._is_usable_detection(detection):
            return None, "not_usable_detection"

        camera_pos, ray_dir = self._camera_ray(sensor_data, detection, image_shape)
        mono_raw = self._monocular_gate_estimate_from_ray(camera_pos, ray_dir, detection)
        mono, mono_reason = self._sanitize_gate_estimate_with_reason(gate_idx, mono_raw)
        if mono is None:
            return None, mono_reason
        return mono, "accepted"

    def _correct_stored_gate_from_vision(self, now, pos, sensor_data, detection, image_shape, gate):
        visual_gate, reason = self._current_visual_gate_estimate(
            self.gate_idx,
            sensor_data,
            detection,
            image_shape,
            require_full=True,
        )
        if visual_gate is None:
            self._log_throttled(
                f"stored_visual_no_correction_{self.gate_idx}",
                now,
                0.75,
                f"stored_visual_no_correction lap={self.route_lap} gate={self.gate_idx} reason={reason} det={self._det_summary(detection)}",
            )
            return gate

        shift = np.linalg.norm(gate[:2] - visual_gate[:2])
        if shift < 0.10:
            return gate

        corrected = visual_gate.copy()
        self.gate_estimates[self.gate_idx] = corrected
        self._update_pass_direction_from_detection(self.gate_idx, sensor_data, detection, image_shape, corrected)
        self._learn_pass_direction(self.gate_idx, corrected, pos)
        self._log(
            f"stored_visual_correction lap={self.route_lap} gate={self.gate_idx} "
            f"old={self._vec(gate)} visual={self._vec(visual_gate)} shift={shift:.2f}"
        )
        return corrected

    def _detection_matches_gate(self, sensor_data, detection, image_shape, gate_idx):
        if not self._is_usable_detection(detection):
            return False

        camera_pos, ray_dir = self._camera_ray(sensor_data, detection, image_shape)
        mono = self._monocular_gate_estimate_from_ray(camera_pos, ray_dir, detection)
        if self._sanitize_gate_estimate(gate_idx, mono) is not None:
            return True

        estimate = self.gate_estimates[gate_idx]
        if estimate is None:
            return False

        to_estimate = estimate - camera_pos
        dist = np.linalg.norm(to_estimate)
        if dist < 1e-6:
            return False
        bearing_alignment = float(np.dot(to_estimate / dist, ray_dir))
        return bearing_alignment > 0.86

    def _detection_is_search_hint(self, sensor_data, detection, image_shape, gate_idx):
        if not self._is_usable_detection(detection):
            return False

        camera_pos, ray_dir = self._camera_ray(sensor_data, detection, image_shape)
        expected = self._expected_angle(gate_idx)
        max_angle_error = 0.75 * self.segment_angular_size

        for depth in np.linspace(0.35, 4.80, 22):
            point = camera_pos + depth * ray_dir
            if point[2] < 0.45 or point[2] > 2.25:
                continue

            rel = point[:2] - self.center
            radius = np.linalg.norm(rel)
            if radius < self.inner_radius - 0.35 or radius > self.outer_radius + 0.45:
                continue

            angle = self._angle_from_position(point[:2])
            if abs(self._angle_diff(angle, expected)) <= max_angle_error:
                return True

        return False

    def _detect_gate(self, camera_data):
        if camera_data is None or camera_data.size == 0:
            return None

        bgr = camera_data[:, :, :3].copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(
            hsv,
            np.array([135, 45, 45], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )

        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        mask_rgb = ((r > 80) & (b > 80) & (g < 135) & (r + b > 2 * g + 70)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask_hsv, mask_rgb)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 45:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 6 or bh < 8:
                continue
            fill = area / float(bw * bh)
            if fill < 0.12:
                continue
            touches_edge = x <= 1 or y <= 1 or x + bw >= w - 2 or y + bh >= h - 2
            moments = cv2.moments(contour)
            if abs(moments["m00"]) > 1e-6:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                cx = x + bw / 2.0
                cy = y + bh / 2.0

            center_penalty = abs((cx - w / 2.0) / (w / 2.0))
            score = area * (0.8 + 0.2 * min(fill, 1.0)) / (1.0 + 0.15 * center_penalty)
            if best is None or score > best["score"]:
                quad = self._quad_from_contour(contour)
                best = {
                    "score": score,
                    "area": float(area),
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "center": (float(cx), float(cy)),
                    "err_x": float((cx - w / 2.0) / (w / 2.0)),
                    "err_y": float((cy - h / 2.0) / (h / 2.0)),
                    "fill": float(fill),
                    "touches_edge": bool(touches_edge),
                    "quad": quad,
                    "mask": mask,
                }
        return best

    def _quad_from_contour(self, contour):
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            return None

        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.035 * perimeter, True)
        if len(approx) == 4:
            points = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rect).astype(np.float32)

        return tuple((float(x), float(y)) for x, y in self._order_image_quad(points))

    def _order_image_quad(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] != 4:
            return points

        by_y = points[np.argsort(points[:, 1])]
        top = by_y[:2][np.argsort(by_y[:2, 0])]
        bottom = by_y[2:][np.argsort(by_y[2:, 0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

    def _update_pass_direction_from_detection(self, gate_idx, sensor_data, detection, image_shape, gate_estimate):
        direction, reason, confidence = self._visual_pass_direction(gate_idx, sensor_data, detection, image_shape, gate_estimate)
        if direction is None:
            self._log_throttled(
                f"visual_pass_direction_skip_{gate_idx}_{reason.split()[0]}",
                float(sensor_data.get("t", 0.0)),
                0.75,
                f"visual_pass_direction_skip gate={gate_idx} reason={reason} det={self._det_summary(detection)}",
            )
            return False

        old = self.gate_pass_directions[gate_idx]
        old_conf = self.gate_pass_direction_confidence[gate_idx]
        if old is None or self.gate_pass_direction_sources[gate_idx] != "vision_pose":
            fused = direction
        elif np.dot(old[:2], direction[:2]) <= 0.35:
            self._log(
                f"visual_pass_direction_reject gate={gate_idx} reason=disagrees_with_stored "
                f"old={self._vec(old)} candidate={self._vec(direction)} confidence={confidence:.2f}"
            )
            return False
        else:
            old_weight = max(old_conf, 0.35)
            new_weight = max(confidence, 0.35)
            fused = old_weight * old + new_weight * direction
            norm = np.linalg.norm(fused[:2])
            if norm < 1e-9:
                return False
            fused = fused / norm

        self.gate_pass_directions[gate_idx] = fused.copy()
        self.gate_pass_direction_sources[gate_idx] = "vision_pose"
        self.gate_pass_direction_confidence[gate_idx] = max(old_conf, confidence)
        self._log(
            f"visual_pass_direction_accept gate={gate_idx} confidence={confidence:.2f} "
            f"reason={reason} direction={self._vec(fused)} gate={self._vec(gate_estimate)} "
            f"det={self._det_summary(detection)}"
        )
        return True

    def _visual_pass_direction(self, gate_idx, sensor_data, detection, image_shape, gate_estimate):
        if not self._is_full_gate_detection(detection):
            return None, "not_full_detection", 0.0
        quad = detection.get("quad")
        if quad is None:
            return None, "missing_quad", 0.0

        image_points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        if not np.all(np.isfinite(image_points)):
            return None, "nonfinite_quad", 0.0

        quad_area = abs(cv2.contourArea(image_points.reshape(-1, 1, 2)))
        if quad_area < 300.0:
            return None, f"quad_too_small area={quad_area:.1f}", 0.0

        height = float(image_shape[0])
        width = float(image_shape[1])
        f_pixels = width / (2.0 * math.tan(self.camera_fov / 2.0))
        camera_matrix = np.array(
            [
                [f_pixels, 0.0, width / 2.0],
                [0.0, f_pixels, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        gate_width = self._estimated_gate_width_from_detection(detection)
        half_width = 0.5 * gate_width
        half_height = 0.5 * self.opening_height
        object_points = np.array(
            [
                [0.0, half_width, half_height],
                [0.0, -half_width, half_height],
                [0.0, -half_width, -half_height],
                [0.0, half_width, -half_height],
            ],
            dtype=np.float64,
        )

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, "solvepnp_failed", 0.0

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        reproj_error = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
        if reproj_error > 22.0:
            return None, f"reprojection_error={reproj_error:.1f}", 0.0

        rotation_obj_to_camera, _ = cv2.Rodrigues(rvec)
        normal_camera = rotation_obj_to_camera[:, 0]
        normal_body = np.array([normal_camera[2], -normal_camera[0], -normal_camera[1]], dtype=float)
        rotation_body_to_world = self._rotation_body_to_world(sensor_data)
        normal_world = rotation_body_to_world @ normal_body
        normal_world[2] = 0.0
        norm = np.linalg.norm(normal_world[:2])
        if norm < 1e-9:
            return None, "horizontal_normal_degenerate", 0.0
        direction = normal_world / norm

        tangent = self._geometric_pass_direction_for_gate(gate_idx, gate_estimate)
        if np.dot(direction[:2], tangent[:2]) < 0.0:
            direction = -direction
        tangent_alignment = float(np.dot(direction[:2], tangent[:2]))
        if tangent_alignment < math.cos(math.radians(55.0)):
            return None, f"too_far_from_nominal align={tangent_alignment:.2f}", 0.0

        confidence = float(np.clip((quad_area / 1200.0) * (22.0 - reproj_error) / 22.0, 0.15, 1.0))
        reason = f"pose_from_quad width={gate_width:.2f} reproj={reproj_error:.1f} align={tangent_alignment:.2f}"
        return direction, reason, confidence

    def _estimated_gate_width_from_detection(self, detection):
        quad = detection.get("quad")
        if quad is None:
            return self.assumed_gate_width

        pts = np.asarray(quad, dtype=float).reshape(4, 2)
        top_width = np.linalg.norm(pts[1] - pts[0])
        bottom_width = np.linalg.norm(pts[2] - pts[3])
        left_height = np.linalg.norm(pts[3] - pts[0])
        right_height = np.linalg.norm(pts[2] - pts[1])
        pixel_height = max(0.5 * (left_height + right_height), 1.0)
        pixel_width = 0.5 * (top_width + bottom_width)
        width = self.opening_height * pixel_width / pixel_height
        return float(np.clip(width, 0.30, 0.50))

    def _camera_ray(self, sensor_data, detection, image_shape):
        h = float(image_shape[0])
        w = float(image_shape[1])
        f_pixels = w / (2.0 * math.tan(self.camera_fov / 2.0))
        cx, cy = detection["center"]
        u = cx - w / 2.0
        v = cy - h / 2.0

        ray_body = np.array([f_pixels, -u, -v], dtype=float)
        ray_body /= np.linalg.norm(ray_body)

        rotation = self._rotation_body_to_world(sensor_data)
        drone_pos = self._position(sensor_data)
        camera_pos = drone_pos + rotation @ self.camera_offset_body
        ray_world = rotation @ ray_body
        ray_world /= np.linalg.norm(ray_world)
        return camera_pos, ray_world

    def _monocular_gate_estimate(self, pos, yaw, detection, image_shape):
        fake_sensor = {
            "x_global": pos[0],
            "y_global": pos[1],
            "z_global": pos[2],
            "q_x": 0.0,
            "q_y": 0.0,
            "q_z": math.sin(yaw / 2.0),
            "q_w": math.cos(yaw / 2.0),
        }
        camera_pos, ray_dir = self._camera_ray(fake_sensor, detection, image_shape)
        return self._monocular_gate_estimate_from_ray(camera_pos, ray_dir, detection)

    def _monocular_gate_estimate_from_ray(self, camera_pos, ray_dir, detection):
        _, _, bw, bh = detection["bbox"]
        image_height = 300.0
        f_pixels = image_height / (2.0 * math.tan(self.camera_fov / 2.0))
        depth_from_height = self.opening_height * f_pixels / max(float(bh), 1.0)
        depth_from_width = self.assumed_gate_width * f_pixels / max(float(bw), 1.0)
        depth = 0.70 * depth_from_height + 0.30 * depth_from_width
        depth = float(np.clip(depth, 0.25, 4.20))
        return camera_pos + depth * ray_dir

    def _triangulate_rays(self, observations):
        if len(observations) < 2:
            return None

        a = np.zeros((3, 3))
        b = np.zeros(3)
        eye = np.eye(3)
        for camera_pos, direction in observations:
            direction = direction / np.linalg.norm(direction)
            projector = eye - np.outer(direction, direction)
            a += projector
            b += projector @ camera_pos

        try:
            point = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            point = np.linalg.lstsq(a, b, rcond=None)[0]

        return point

    def _evaluate_gate_estimate_quality(self, gate_idx, estimate, source, detection, observations, samples_after):
        baseline = self._ray_observation_baseline(observations)
        residual = self._ray_residual(estimate, observations)
        recent_spread = self._recent_estimate_spread(gate_idx, estimate)
        full_detection = self._is_full_gate_detection(detection)

        checks = [
            ("samples", samples_after >= self.min_reliable_estimate_samples),
            ("observations", len(observations) >= self.min_reliable_observations),
            ("baseline", baseline >= self.min_reliable_baseline),
            ("ray_residual", residual <= self.max_reliable_ray_residual),
            ("recent_spread", recent_spread <= self.max_reliable_recent_spread),
            ("full_detection", full_detection),
        ]
        failed = [name for name, ok in checks if not ok]
        ok = not failed
        return {
            "ok": ok,
            "status": "ok" if ok else "not_ready",
            "reason": "ok" if ok else "+".join(failed),
            "source": source,
            "samples": samples_after,
            "observations": len(observations),
            "baseline": baseline,
            "ray_residual": residual,
            "recent_spread": recent_spread,
            "full_detection": full_detection,
        }

    def _estimate_quality_ready_for_commit(self, gate_idx):
        quality = self.gate_estimate_quality[gate_idx]
        if quality is None:
            return False, "missing_quality", None
        if not quality["ok"]:
            return False, quality["reason"], quality
        return True, "ok", quality

    def _ray_observation_baseline(self, observations):
        if len(observations) < 2:
            return 0.0

        max_distance = 0.0
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                distance = float(np.linalg.norm(observations[i][0] - observations[j][0]))
                max_distance = max(max_distance, distance)
        return max_distance

    def _ray_residual(self, point, observations):
        if point is None or len(observations) < 2:
            return float("inf")

        point = np.asarray(point, dtype=float)
        distances = []
        for camera_pos, direction in observations:
            direction = direction / max(np.linalg.norm(direction), 1e-9)
            delta = point - camera_pos
            perpendicular = delta - float(np.dot(delta, direction)) * direction
            distances.append(float(np.linalg.norm(perpendicular)))
        if not distances:
            return float("inf")
        return float(np.mean(distances))

    def _recent_estimate_spread(self, gate_idx, estimate):
        recent = self.estimate_history[gate_idx][-4:] + [np.asarray(estimate, dtype=float).copy()]
        if len(recent) < 3:
            return float("inf")

        points = np.asarray(recent, dtype=float)
        center = np.mean(points, axis=0)
        return float(np.max(np.linalg.norm(points - center, axis=1)))

    def _sanitize_gate_estimate(self, gate_idx, estimate):
        sanitized, _ = self._sanitize_gate_estimate_with_reason(gate_idx, estimate)
        return sanitized

    def _sanitize_gate_estimate_with_reason(self, gate_idx, estimate):
        if estimate is None or not np.all(np.isfinite(estimate)):
            return None, "missing_or_nonfinite"

        estimate = np.asarray(estimate, dtype=float).copy()
        rel = estimate[:2] - self.center
        radius = np.linalg.norm(rel)
        if radius < 1e-6:
            return None, "at_course_center"

        expected = self._expected_angle(gate_idx)
        angle = self._angle_from_position(estimate[:2])
        if abs(self._angle_diff(angle, expected)) > 0.34:
            return None, f"wrong_angle angle={angle:.3f} expected={expected:.3f}"
        if self._segment_from_position(estimate[:2]) != gate_idx + 1:
            return None, f"wrong_segment segment={self._segment_from_position(estimate[:2])} expected={gate_idx + 1}"
        if radius < self.inner_radius - 0.45 or radius > self.outer_radius + 0.55:
            return None, f"radius_out_of_range radius={radius:.2f}"
        for prev_idx in range(gate_idx):
            prev = self.gate_estimates[prev_idx]
            if prev is not None and np.linalg.norm(estimate[:2] - prev[:2]) < 0.72:
                return None, f"too_close_to_previous_gate prev={prev_idx}"

        estimate[0] = np.clip(estimate[0], 0.25, 7.75)
        estimate[1] = np.clip(estimate[1], 0.25, 7.75)
        estimate[2] = np.clip(estimate[2], 0.70, 2.05)
        return estimate, "accepted"

    def _limited_setpoint(self, pos, target, max_step, yaw_target):
        target = np.asarray(target, dtype=float).copy()
        target[0] = np.clip(target[0], 0.25, 7.75)
        target[1] = np.clip(target[1], 0.25, 7.75)
        target[2] = np.clip(target[2], 0.65, 2.10)

        delta_xy = target[:2] - pos[:2]
        dist_xy = np.linalg.norm(delta_xy)
        if dist_xy > max_step and dist_xy > 1e-9:
            cmd_xy = pos[:2] + delta_xy / dist_xy * max_step
        else:
            cmd_xy = target[:2]

        z_delta = target[2] - pos[2]
        z_cmd = pos[2] + np.clip(z_delta, -0.22, 0.22)
        return [float(cmd_xy[0]), float(cmd_xy[1]), float(z_cmd), float(self._wrap_angle(yaw_target))]

    def _build_stored_replay_plan(self, now, pos):
        last_plan = None
        for attempt_idx, (attempt_name, entry_distance, exit_distance, time_multiplier) in enumerate(
            self.replay_plan_attempts
        ):
            plan = self._build_stored_replay_plan_attempt(
                now,
                pos,
                attempt_name,
                entry_distance,
                exit_distance,
                time_multiplier,
            )
            last_plan = plan
            validation = plan["validation"]
            if validation["ok"]:
                if attempt_idx > 0:
                    self._log(
                        f"stored_replay_replan_selected lap={self.route_lap} attempt={attempt_name} "
                        f"reason=previous_validation_failed status={validation['status']} "
                        f"duration={plan['times'][-1]:.2f}"
                    )
                return plan

            next_attempt = (
                self.replay_plan_attempts[attempt_idx + 1][0]
                if attempt_idx + 1 < len(self.replay_plan_attempts)
                else "none"
            )
            self._log(
                f"stored_replay_replan lap={self.route_lap} attempt={attempt_name} "
                f"status={validation['status']} reason={validation['reason']} next={next_attempt}"
            )

        self._log(
            f"stored_replay_replan_exhausted lap={self.route_lap} "
            f"selected_attempt={last_plan['attempt_name']} status={last_plan['validation']['status']} "
            f"reason={last_plan['validation']['reason']}"
        )
        return last_plan

    def _build_stored_replay_plan_attempt(self, now, pos, attempt_name, entry_distance, exit_distance, time_multiplier):
        waypoints = [np.asarray(pos, dtype=float).copy()]
        gates = []
        for gate_idx, gate_estimate in enumerate(self.gate_estimates):
            entry, center, exit_point, pass_dir, pass_yaw = self._gate_waypoints(
                gate_idx,
                gate_estimate,
                entry_distance=entry_distance,
                exit_distance=exit_distance,
            )
            entry_idx = len(waypoints)
            center_idx = entry_idx + 1
            exit_idx = entry_idx + 2
            waypoints.extend([entry, center, exit_point])
            gates.append(
                {
                    "gate_idx": gate_idx,
                    "gate": center.copy(),
                    "entry": entry.copy(),
                    "center": center.copy(),
                    "exit": exit_point.copy(),
                    "pass_dir": pass_dir.copy(),
                    "pass_yaw": pass_yaw,
                    "entry_idx": entry_idx,
                    "center_idx": center_idx,
                    "exit_idx": exit_idx,
                    "done": False,
                }
            )

        waypoints.append(self.takeoff_pos.copy())
        waypoints = np.asarray(waypoints, dtype=float)
        times = self._stored_replay_times(waypoints) * time_multiplier
        coeffs = self._minimum_jerk_coefficients(waypoints, times)
        metrics = self._stored_replay_metrics(coeffs, times)

        target_speed = 0.98 * self.replay_speed_limit
        target_accel = 0.98 * self.replay_accel_limit
        scale = max(
            1.0,
            metrics["max_speed"] / target_speed,
            math.sqrt(metrics["max_accel"] / target_accel),
        )
        if scale > 1.0:
            times = times * scale
            coeffs = self._minimum_jerk_coefficients(waypoints, times)
            metrics = self._stored_replay_metrics(coeffs, times)
        total_time_scale = time_multiplier * scale

        for gate in gates:
            gate["entry_time"] = float(times[gate["entry_idx"]])
            gate["center_time"] = float(times[gate["center_idx"]])
            gate["exit_time"] = float(times[gate["exit_idx"]])

        plan = {
            "start_time": now,
            "waypoints": waypoints,
            "times": times,
            "coeffs": coeffs,
            "gates": gates,
            "active_gate": 0,
            "metrics": metrics,
            "attempt_name": attempt_name,
            "entry_distance": entry_distance,
            "exit_distance": exit_distance,
            "time_multiplier": time_multiplier,
        }
        validation = self._validate_stored_replay_plan(plan)
        plan["validation"] = validation
        self._log(
            f"stored_replay_plan lap={self.route_lap} attempt={attempt_name} waypoints={len(waypoints)} "
            f"duration={times[-1]:.2f} max_speed={metrics['max_speed']:.3f} "
            f"max_accel={metrics['max_accel']:.3f} time_scale={total_time_scale:.2f} "
            f"entry_distance={entry_distance:.2f} exit_distance={exit_distance:.2f} "
            f"validation={validation['status']}"
        )
        return plan

    def _stored_replay_times(self, waypoints):
        times = [0.0]
        for idx in range(1, len(waypoints)):
            distance = float(np.linalg.norm(waypoints[idx] - waypoints[idx - 1]))
            segment_time = float(np.clip(distance / 0.72, 0.80, 3.80))
            times.append(times[-1] + segment_time)
        return np.asarray(times, dtype=float)

    def _minimum_jerk_poly_matrix(self, t):
        return np.array(
            [
                [t**5, t**4, t**3, t**2, t, 1.0],
                [5.0 * t**4, 4.0 * t**3, 3.0 * t**2, 2.0 * t, 1.0, 0.0],
                [20.0 * t**3, 12.0 * t**2, 6.0 * t, 2.0, 0.0, 0.0],
                [60.0 * t**2, 24.0 * t, 6.0, 0.0, 0.0, 0.0],
                [120.0 * t, 24.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def _minimum_jerk_coefficients(self, waypoints, times):
        seg_times = np.diff(times)
        segment_count = len(waypoints) - 1
        coeffs = np.zeros((6 * segment_count, 3), dtype=float)
        A_0 = self._minimum_jerk_poly_matrix(0.0)

        for dim in range(3):
            A = np.zeros((6 * segment_count, 6 * segment_count), dtype=float)
            b = np.zeros(6 * segment_count, dtype=float)
            row = 0
            pos = waypoints[:, dim]
            for idx in range(segment_count):
                A_f = self._minimum_jerk_poly_matrix(float(seg_times[idx]))
                A[row, idx * 6 : (idx + 1) * 6] = A_0[0]
                b[row] = pos[idx]
                row += 1
                A[row, idx * 6 : (idx + 1) * 6] = A_f[0]
                b[row] = pos[idx + 1]
                row += 1

                if idx == 0:
                    A[row, idx * 6 : (idx + 1) * 6] = A_0[1]
                    row += 1
                    A[row, idx * 6 : (idx + 1) * 6] = A_0[2]
                    row += 1

                if idx < segment_count - 1:
                    A[row : row + 4, idx * 6 : (idx + 1) * 6] = A_f[1:]
                    A[row : row + 4, (idx + 1) * 6 : (idx + 2) * 6] = -A_0[1:]
                    row += 4
                else:
                    A[row, idx * 6 : (idx + 1) * 6] = A_f[1]
                    row += 1
                    A[row, idx * 6 : (idx + 1) * 6] = A_f[2]
                    row += 1

            coeffs[:, dim] = np.linalg.solve(A, b)

        return coeffs

    def _sample_stored_replay(self, plan, elapsed):
        target, velocity, acceleration = self._sample_stored_replay_raw(
            plan["coeffs"],
            plan["times"],
            elapsed,
        )
        target = target.copy()
        target[0] = np.clip(target[0], 0.25, 7.75)
        target[1] = np.clip(target[1], 0.25, 7.75)
        target[2] = np.clip(target[2], 0.65, 2.10)
        return target, velocity, acceleration

    def _sample_stored_replay_raw(self, coeffs, times, elapsed):
        t = float(np.clip(elapsed, 0.0, times[-1]))
        seg_idx = min(max(np.searchsorted(times, t, side="right") - 1, 0), len(times) - 2)
        local_t = t - times[seg_idx]
        basis = self._minimum_jerk_poly_matrix(local_t)
        seg_coeffs = coeffs[seg_idx * 6 : (seg_idx + 1) * 6]
        target = basis[0] @ seg_coeffs
        velocity = basis[1] @ seg_coeffs
        acceleration = basis[2] @ seg_coeffs
        return target, velocity, acceleration

    def _stored_replay_metrics(self, coeffs, times):
        max_speed = 0.0
        max_accel = 0.0
        sample_count = max(120, int(math.ceil(times[-1] / 0.05)))
        for t in np.linspace(0.0, times[-1], sample_count):
            _, velocity, acceleration = self._sample_stored_replay_raw(coeffs, times, float(t))
            max_speed = max(max_speed, float(np.linalg.norm(velocity)))
            max_accel = max(max_accel, float(np.linalg.norm(acceleration)))
        return {"max_speed": max_speed, "max_accel": max_accel}

    def _validate_stored_replay_plan(self, plan):
        attempt_name = plan.get("attempt_name", "unknown")
        coeffs = plan["coeffs"]
        times = plan["times"]
        waypoints = plan["waypoints"]
        bounds_min = np.array([0.25, 0.25, 0.65], dtype=float)
        bounds_max = np.array([7.75, 7.75, 2.10], dtype=float)

        max_waypoint_error = 0.0
        for waypoint, waypoint_time in zip(waypoints, times):
            target, _, _ = self._sample_stored_replay_raw(coeffs, times, float(waypoint_time))
            max_waypoint_error = max(max_waypoint_error, float(np.linalg.norm(target - waypoint)))

        sample_count = max(120, int(math.ceil(float(times[-1]) / 0.05)))
        max_speed = 0.0
        max_accel = 0.0
        min_bounds_margin = float("inf")
        for sample_t in np.linspace(0.0, float(times[-1]), sample_count):
            target, velocity, acceleration = self._sample_stored_replay_raw(coeffs, times, float(sample_t))
            max_speed = max(max_speed, float(np.linalg.norm(velocity)))
            max_accel = max(max_accel, float(np.linalg.norm(acceleration)))
            min_bounds_margin = min(
                min_bounds_margin,
                float(np.min(target - bounds_min)),
                float(np.min(bounds_max - target)),
            )

        min_width_margin = float("inf")
        min_height_margin = float("inf")
        gate_ok = True
        for gate in plan["gates"]:
            crossing, crossing_found = self._planned_gate_crossing(plan, gate)
            center_sample, _, _ = self._sample_stored_replay_raw(coeffs, times, gate["center_time"])
            center_error = float(np.linalg.norm(center_sample - gate["center"]))
            lateral = self._gate_lateral_error(crossing, gate["gate"], gate["pass_dir"])
            vertical = float(crossing[2] - gate["gate"][2])
            width_margin = 0.15 - abs(lateral)
            height_margin = 0.20 - abs(vertical)
            min_width_margin = min(min_width_margin, width_margin)
            min_height_margin = min(min_height_margin, height_margin)
            this_gate_ok = (
                crossing_found
                and center_error <= self.replay_waypoint_tolerance
                and width_margin >= self.replay_min_width_margin
                and height_margin >= self.replay_min_height_margin
            )
            gate_ok = gate_ok and this_gate_ok
            self._log(
                f"stored_replay_validation_gate lap={self.route_lap} attempt={attempt_name} gate={gate['gate_idx']} "
                f"status={'ok' if this_gate_ok else 'failed'} crossing_found={crossing_found} "
                f"crossing={self._vec(crossing)} center_error={center_error:.4f} "
                f"lateral_error={lateral:.3f} vertical_error={vertical:.3f} "
                f"conservative_width_margin={width_margin:.3f} height_margin={height_margin:.3f}"
            )

        speed_ok = max_speed <= self.replay_speed_limit + 1e-6
        accel_ok = max_accel <= self.replay_accel_limit + 1e-6
        waypoint_ok = max_waypoint_error <= self.replay_waypoint_tolerance
        bounds_ok = min_bounds_margin >= self.replay_min_bounds_margin - 1e-6
        ok = gate_ok and speed_ok and accel_ok and waypoint_ok and bounds_ok
        status = "ok" if ok else "failed"
        failure_reasons = []
        if not waypoint_ok:
            failure_reasons.append("waypoint")
        if not gate_ok:
            failure_reasons.append("gate_margin")
        if not speed_ok:
            failure_reasons.append("speed")
        if not accel_ok:
            failure_reasons.append("accel")
        if not bounds_ok:
            failure_reasons.append("bounds")
        reason = "ok" if ok else "+".join(failure_reasons)
        self._log(
            f"stored_replay_validation lap={self.route_lap} attempt={attempt_name} status={status} "
            f"max_waypoint_error={max_waypoint_error:.5f} waypoint_ok={waypoint_ok} "
            f"min_width_margin={min_width_margin:.3f} min_height_margin={min_height_margin:.3f} gate_ok={gate_ok} "
            f"max_speed={max_speed:.3f} speed_limit={self.replay_speed_limit:.3f} speed_ok={speed_ok} "
            f"max_accel={max_accel:.3f} accel_limit={self.replay_accel_limit:.3f} accel_ok={accel_ok} "
            f"min_bounds_margin={min_bounds_margin:.3f} bounds_limit={self.replay_min_bounds_margin:.3f} "
            f"bounds_ok={bounds_ok} samples={sample_count} reason={reason}"
        )
        return {
            "ok": ok,
            "status": status,
            "reason": reason,
            "max_waypoint_error": max_waypoint_error,
            "min_width_margin": min_width_margin,
            "min_height_margin": min_height_margin,
            "max_speed": max_speed,
            "max_accel": max_accel,
            "min_bounds_margin": min_bounds_margin,
        }

    def _planned_gate_crossing(self, plan, gate):
        start_t = gate["entry_time"]
        end_t = gate["exit_time"]
        sample_count = 48
        previous_pos = None
        previous_progress = None
        for sample_t in np.linspace(start_t, end_t, sample_count):
            target, _, _ = self._sample_stored_replay_raw(plan["coeffs"], plan["times"], float(sample_t))
            progress = float(np.dot(target[:2] - gate["gate"][:2], gate["pass_dir"][:2]))
            if previous_pos is not None and previous_progress is not None:
                if previous_progress <= 0.0 <= progress or previous_progress >= 0.0 >= progress:
                    denom = progress - previous_progress
                    alpha = 1.0 if abs(denom) < 1e-9 else -previous_progress / denom
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    return previous_pos + alpha * (target - previous_pos), True
            previous_pos = target
            previous_progress = progress

        center_target, _, _ = self._sample_stored_replay_raw(plan["coeffs"], plan["times"], gate["center_time"])
        return center_target, False

    def _update_stored_replay_progress(self, now, pos, plan, elapsed):
        active = int(plan["active_gate"])
        if active >= self.num_gates:
            self.gate_idx = self.num_gates
            return

        gate = plan["gates"][active]
        self.gate_idx = active
        self._log_stored_crossing_margin(
            now,
            pos,
            gate["gate"],
            gate["entry"],
            gate["center"],
            gate["exit"],
            gate["pass_dir"],
        )
        progress = float(np.dot(pos[:2] - gate["gate"][:2], gate["pass_dir"][:2]))
        exit_error = float(np.linalg.norm(pos - gate["exit"]))
        segment = self._segment_from_position(pos[:2])
        crossed_expected_segment = segment == active + 1
        done_reason = None
        if progress > 0.76:
            done_reason = "clear_progress"
        elif crossed_expected_segment and progress > 0.42:
            done_reason = "next_segment"
        elif exit_error < 0.28 and progress > 0.50:
            done_reason = "near_exit"

        if done_reason is not None:
            if not self.stored_crossing_logged:
                self._log_stored_crossing_sample(
                    "done_before_crossing_log",
                    self.route_lap,
                    active,
                    pos,
                    gate["gate"],
                    gate["pass_dir"],
                    progress,
                )
            self._log(
                f"stored_replay_gate_done lap={self.route_lap} gate={active} "
                f"reason={done_reason} progress={progress:.2f} "
                f"exit_error={exit_error:.2f} segment={segment} replay_t={elapsed:.2f}"
            )
            plan["active_gate"] = active + 1
            self.gate_idx = active + 1
            self.stored_crossing_key = None

    def _smooth_stored_setpoint(self, now, pos, target, yaw_target, key):
        target = np.asarray(target, dtype=float).copy()
        target[0] = np.clip(target[0], 0.25, 7.75)
        target[1] = np.clip(target[1], 0.25, 7.75)
        target[2] = np.clip(target[2], 0.65, 2.10)

        if self.stored_curve_key != key or self.stored_curve_start is None:
            self.stored_curve_key = key
            self.stored_curve_started_at = now
            self.stored_curve_start = np.asarray(pos, dtype=float).copy()
            distance = float(np.linalg.norm(target - self.stored_curve_start))
            self.stored_curve_duration = float(np.clip(distance / 0.70, 0.85, 3.20))
            peak_speed = 1.875 * distance / max(self.stored_curve_duration, 1e-6)
            peak_accel = (10.0 * math.sqrt(3.0) / 3.0) * distance / max(self.stored_curve_duration**2, 1e-6)
            self._log(
                f"stored_curve_start key={key} start={self._vec(self.stored_curve_start)} "
                f"target={self._vec(target)} duration={self.stored_curve_duration:.2f} "
                f"expected_peak_speed={peak_speed:.3f} expected_peak_accel={peak_accel:.3f}"
            )

        elapsed = max(0.0, now - self.stored_curve_started_at)
        duration = max(self.stored_curve_duration, 1e-6)
        s = float(np.clip(elapsed / duration, 0.0, 1.0))
        blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        delta = target - self.stored_curve_start
        command = self.stored_curve_start + blend * delta
        blend_rate = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
        blend_accel = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (duration * duration)
        speed = float(np.linalg.norm(blend_rate * delta))
        accel = float(np.linalg.norm(blend_accel * delta))
        self._record_stored_motion_metrics(speed, accel, key, s)
        return [float(command[0]), float(command[1]), float(command[2]), float(self._wrap_angle(yaw_target))]

    def _record_stored_motion_metrics(self, speed, accel, key, s):
        context = f"lap={self.route_lap} gate={self.gate_idx} key={key} s={s:.2f}"
        if speed > self.stored_motion_lap_max_speed:
            self.stored_motion_lap_max_speed = speed
            self.stored_motion_lap_max_speed_context = context
        if accel > self.stored_motion_lap_max_accel:
            self.stored_motion_lap_max_accel = accel
            self.stored_motion_lap_max_accel_context = context
        self.stored_motion_run_max_speed = max(self.stored_motion_run_max_speed, speed)
        self.stored_motion_run_max_accel = max(self.stored_motion_run_max_accel, accel)

    def _log_stored_motion_summary(self, lap, reason):
        self._log(
            f"stored_motion_summary reason={reason} lap={lap} "
            f"max_speed={self.stored_motion_lap_max_speed:.3f} "
            f"max_accel={self.stored_motion_lap_max_accel:.3f} "
            f"speed_context={self.stored_motion_lap_max_speed_context} "
            f"accel_context={self.stored_motion_lap_max_accel_context} "
            f"run_max_speed={self.stored_motion_run_max_speed:.3f} "
            f"run_max_accel={self.stored_motion_run_max_accel:.3f}"
        )

    def _log_stored_motion_summary_once(self, lap, reason):
        if self.stored_motion_summary_lap == lap:
            return
        self.stored_motion_summary_lap = lap
        self._log_stored_motion_summary(lap, reason)

    def _reset_stored_motion_lap_metrics(self):
        self.stored_motion_lap_max_speed = 0.0
        self.stored_motion_lap_max_accel = 0.0
        self.stored_motion_lap_max_speed_context = "none"
        self.stored_motion_lap_max_accel_context = "none"

    def _log_stored_crossing_margin(self, now, pos, gate, entry, center, exit_point, pass_dir):
        key = (self.route_lap, self.gate_idx)
        progress = float(np.dot(pos[:2] - gate[:2], pass_dir[:2]))
        if self.stored_crossing_key != key:
            self.stored_crossing_key = key
            self.stored_crossing_prev_pos = None
            self.stored_crossing_prev_progress = None
            self.stored_crossing_logged = False
            planned_lateral = self._gate_lateral_error(center, gate, pass_dir)
            self._log(
                f"stored_plan_margin lap={self.route_lap} gate={self.gate_idx} "
                f"entry={self._vec(entry)} center={self._vec(center)} exit={self._vec(exit_point)} "
                f"planned_lateral={planned_lateral:.3f} conservative_half_width=0.150 half_height=0.200"
            )

        if (
            not self.stored_crossing_logged
            and self.stored_crossing_prev_pos is not None
            and self.stored_crossing_prev_progress is not None
            and self.stored_crossing_prev_progress <= 0.0
            and progress >= 0.0
        ):
            denom = progress - self.stored_crossing_prev_progress
            alpha = 1.0 if abs(denom) < 1e-9 else -self.stored_crossing_prev_progress / denom
            alpha = float(np.clip(alpha, 0.0, 1.0))
            crossing = self.stored_crossing_prev_pos + alpha * (pos - self.stored_crossing_prev_pos)
            self._log_stored_crossing_sample(
                "plane_crossing",
                self.route_lap,
                self.gate_idx,
                crossing,
                gate,
                pass_dir,
                0.0,
            )
            self.stored_crossing_logged = True

        self.stored_crossing_prev_pos = pos.copy()
        self.stored_crossing_prev_progress = progress

    def _log_stored_crossing_sample(self, reason, lap, gate_idx, sample_pos, gate, pass_dir, progress):
        lateral = self._gate_lateral_error(sample_pos, gate, pass_dir)
        vertical = float(sample_pos[2] - gate[2])
        center_error = math.sqrt(lateral * lateral + vertical * vertical)
        width_margin = 0.15 - abs(lateral)
        height_margin = 0.20 - abs(vertical)
        self._log(
            f"stored_crossing_margin reason={reason} lap={lap} gate={gate_idx} "
            f"sample={self._vec(sample_pos)} gate={self._vec(gate)} "
            f"lateral_error={lateral:.3f} vertical_error={vertical:.3f} center_error={center_error:.3f} "
            f"conservative_width_margin={width_margin:.3f} height_margin={height_margin:.3f} "
            f"progress={progress:.3f}"
        )

    def _gate_lateral_error(self, sample_pos, gate, pass_dir):
        right = np.array([-pass_dir[1], pass_dir[0]], dtype=float)
        norm = np.linalg.norm(right)
        if norm < 1e-9:
            return 0.0
        right /= norm
        return float(np.dot(np.asarray(sample_pos)[:2] - np.asarray(gate)[:2], right))

    def _position(self, sensor_data):
        return np.array(
            [
                float(sensor_data["x_global"]),
                float(sensor_data["y_global"]),
                float(sensor_data["z_global"]),
            ]
        )

    def _rotation_body_to_world(self, sensor_data):
        x = float(sensor_data["q_x"])
        y = float(sensor_data["q_y"])
        z = float(sensor_data["q_z"])
        w = float(sensor_data["q_w"])
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-9:
            return np.eye(3)
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _expected_angle(self, gate_idx):
        return ((gate_idx + 1) * math.pi / 3.0) % (2.0 * math.pi)

    def _expected_gate_position(self, gate_idx):
        angle = self._expected_angle(gate_idx)
        return self._gate_position_from_angle_radius(angle, self.nominal_radius, 1.25)

    def _gate_position_from_angle_radius(self, angle, radius, height):
        return np.array(
            [
                self.center[0] - radius * math.cos(angle),
                self.center[1] - radius * math.sin(angle),
                height,
            ]
        )

    def _geometric_pass_direction_for_gate(self, gate_idx, gate_estimate=None):
        if gate_estimate is None:
            angle = self._expected_angle(gate_idx)
        else:
            angle = self._angle_from_position(np.asarray(gate_estimate)[:2])
        direction = np.array([math.sin(angle), -math.cos(angle), 0.0])
        norm = np.linalg.norm(direction[:2])
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return direction / norm

    def _pass_direction_for_gate(self, gate_idx, gate_estimate=None, pos=None):
        learned = self.gate_pass_directions[gate_idx]
        if learned is not None:
            return learned.copy()

        return self._geometric_pass_direction_for_gate(gate_idx, gate_estimate)

    def _learn_pass_direction(self, gate_idx, gate_estimate, pos):
        old = self.gate_pass_directions[gate_idx]
        if old is not None and self.gate_pass_direction_sources[gate_idx] == "vision_pose":
            self._log(
                f"pass_direction_kept gate={gate_idx} source=vision_pose "
                f"confidence={self.gate_pass_direction_confidence[gate_idx]:.2f} direction={self._vec(old)}"
            )
            return old.copy()

        tangent = self._geometric_pass_direction_for_gate(gate_idx, gate_estimate)
        direction = tangent
        source = "tangent_fallback"
        if old is not None and np.dot(old[:2], direction[:2]) > 0.0:
            direction = 0.65 * old + 0.35 * direction
            direction /= np.linalg.norm(direction[:2])

        self.gate_pass_directions[gate_idx] = direction.copy()
        self.gate_pass_direction_sources[gate_idx] = source
        self.gate_pass_direction_confidence[gate_idx] = max(self.gate_pass_direction_confidence[gate_idx], 0.25)
        self._log(
            f"pass_direction_learned gate={gate_idx} source={source} "
            f"gate={self._vec(gate_estimate)} pos={self._vec(pos)} "
            f"tangent={self._vec(tangent)} direction={self._vec(direction)}"
        )
        return direction

    def _gate_waypoints(self, gate_idx, gate_estimate, entry_distance=0.58, exit_distance=0.88):
        gate = np.asarray(gate_estimate, dtype=float).copy()
        pass_dir = self._pass_direction_for_gate(gate_idx, gate)
        pass_dir[2] = 0.0
        norm = np.linalg.norm(pass_dir[:2])
        if norm < 1e-9:
            pass_dir = self._geometric_pass_direction_for_gate(gate_idx, gate)
        else:
            pass_dir = pass_dir / norm

        center = gate.copy()
        entry = gate - entry_distance * pass_dir
        exit_point = gate + exit_distance * pass_dir
        for point in (entry, center, exit_point):
            point[2] = np.clip(gate[2], 0.74, 2.05)

        pass_yaw = math.atan2(pass_dir[1], pass_dir[0])
        return entry, center, exit_point, pass_dir, pass_yaw

    def _is_usable_detection(self, detection):
        if detection is None:
            return False
        x, y, bw, bh = detection["bbox"]
        return (
            detection["area"] >= 90
            and bw >= 8
            and bh >= 14
            and detection.get("fill", 0.0) >= 0.12
        )

    def _is_full_gate_detection(self, detection):
        if not self._is_usable_detection(detection):
            return False
        _, _, bw, bh = detection["bbox"]
        aspect = bw / max(float(bh), 1.0)
        return (
            not detection.get("touches_edge", False)
            and detection["area"] >= 420
            and bw >= 18
            and bh >= 24
            and 0.22 <= aspect <= 2.4
        )

    def _expected_pass_yaw(self, gate_idx):
        direction = self._pass_direction_for_gate(gate_idx)
        return math.atan2(direction[1], direction[0])

    def _segment_from_position(self, pos_xy):
        rel = np.asarray(pos_xy, dtype=float) - self.center
        norm = np.linalg.norm(rel)
        if norm < 1e-6:
            return -1
        angle = self._angle_from_position(pos_xy)
        for i in range(self.num_segments):
            lo = ((2 * i - 0.5) * self.segment_angular_size) % (2 * math.pi)
            hi = ((2 * i + 0.5) * self.segment_angular_size) % (2 * math.pi)
            if i == 0:
                if angle >= lo or angle <= hi:
                    return i
            elif lo <= angle <= hi:
                return i
        return -1

    def _angle_from_position(self, pos_xy):
        rel = np.asarray(pos_xy, dtype=float) - self.center
        return (math.atan2(rel[1], rel[0]) + math.pi) % (2.0 * math.pi)

    def _angle_diff(self, a, b):
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    def _yaw_to_point(self, pos, target):
        delta = np.asarray(target)[:2] - np.asarray(pos)[:2]
        if np.linalg.norm(delta) < 1e-9:
            return 0.0
        return math.atan2(delta[1], delta[0])

    def _wrap_angle(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _log_status(self, now, pos, yaw, detection, command):
        state_tuple = (
            self.route_lap,
            self.gate_idx,
            self.phase,
            self.stage,
            self.pass_stage,
            self.stored_stage,
            self.commit_gate_idx,
        )
        if state_tuple != self.last_state_tuple:
            self._log(
                f"state_change t={now:.2f} own_lap={self.route_lap} gate={self.gate_idx} "
                f"phase={self.phase} stage={self.stage} pass_stage={self.pass_stage} "
                f"stored_stage={self.stored_stage} commit_gate={self.commit_gate_idx} "
                f"pos={self._vec(pos)} yaw={yaw:.3f}"
            )
            self.last_state_tuple = state_tuple

        if not self.debug_enabled or now - self.last_log_time < 0.5:
            return
        self.last_log_time = now
        estimates = "".join("." if p is None else "x" for p in self.gate_estimates)
        self._log(
            f"t={now:.2f} phase={self.phase} stage={self.stage} own_lap={self.route_lap} "
            f"gate={self.gate_idx} seg={self._segment_from_position(pos[:2])} "
            f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) yaw={yaw:.2f} "
            f"cmd=({command[0]:.2f},{command[1]:.2f},{command[2]:.2f},{command[3]:.2f}) "
            f"det={self._det_summary(detection)} estimates={estimates} samples={self.estimate_samples} "
            f"pass_dirs={self._pass_dir_summary()}"
        )

    def _save_debug_image(self, camera_data, detection, now):
        if not self.debug_enabled or not self.save_images:
            return
        if camera_data is None or now - self.last_image_time < 0.9 or self.image_count >= 180:
            return
        self.last_image_time = now
        self.image_count += 1

        image = camera_data[:, :, :3].copy()
        x, y, w, h = detection["bbox"]
        cx, cy = detection["center"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(image, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        cv2.line(image, (150, 0), (150, 300), (255, 255, 255), 1)
        cv2.line(image, (0, 150), (300, 150), (255, 255, 255), 1)
        cv2.imwrite(str(self.debug_dir / f"camera_{self.image_count:03d}_{now:.1f}.png"), image)
        cv2.imwrite(str(self.debug_dir / f"mask_{self.image_count:03d}_{now:.1f}.png"), detection["mask"])
        self._log(
            f"debug_image_saved index={self.image_count} t={now:.2f} "
            f"camera=camera_{self.image_count:03d}_{now:.1f}.png mask=mask_{self.image_count:03d}_{now:.1f}.png "
            f"det={self._det_summary(detection)}"
        )

    def _vec(self, value):
        if value is None:
            return "None"
        arr = np.asarray(value, dtype=float).reshape(-1)
        return "(" + ",".join(f"{x:.3f}" for x in arr[:3]) + ")"

    def _det_summary(self, detection):
        if detection is None:
            return "none"
        x, y, bw, bh = detection["bbox"]
        return (
            f"area={detection['area']:.0f} score={detection.get('score', 0.0):.0f} "
            f"err=({detection['err_x']:.2f},{detection['err_y']:.2f}) "
            f"bbox=({x},{y},{bw},{bh}) fill={detection.get('fill', 0.0):.2f} "
            f"edge={detection.get('touches_edge', False)}"
        )

    def _quality_summary(self, quality):
        if quality is None:
            return "quality=missing"
        return (
            f"quality={quality['status']} samples={quality['samples']} obs={quality['observations']} "
            f"baseline={quality['baseline']:.2f} residual={quality['ray_residual']:.2f} "
            f"spread={quality['recent_spread']:.2f} full={quality['full_detection']}"
        )

    def _pass_dir_summary(self):
        parts = []
        for idx, direction in enumerate(self.gate_pass_directions):
            if direction is None:
                parts.append(f"{idx}:.")
            else:
                parts.append(f"{idx}:({direction[0]:.2f},{direction[1]:.2f})")
        return "[" + " ".join(parts) + "]"

    def _log_throttled(self, key, now, interval, message):
        if not self.debug_enabled and not self.debug_stdout:
            return
        if not np.isfinite(now):
            now = time.time() - self.start_wall
        last = self.last_throttled_log.get(key, -1e9)
        if now - last < interval:
            return
        self.last_throttled_log[key] = now
        self._log(message)

    def _log(self, message):
        self.log_seq += 1
        line = f"seq={self.log_seq:06d} run_id={self.run_id} layout_id={self.layout_id} {message}"
        if self.debug_stdout:
            print("ARDBG", line, flush=True)
        if not self.debug_enabled:
            return
        line = f"[{time.time() - self.start_wall:8.3f}] {line}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


_controller = MyAssignment()


def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
