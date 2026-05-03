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

        self.gate_estimates = [None] * self.num_gates
        self.gate_pass_directions = [None] * self.num_gates
        self.estimate_samples = [0] * self.num_gates
        self.ray_observations = [[] for _ in range(self.num_gates)]

        self.search_started_at = 0.0
        self.search_gate_idx = None
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
        if (
            detection is not None
            and self.gate_idx < self.num_gates
            and (not self.started_course or not self._detection_matches_gate(sensor_data, detection, camera_data.shape, self.gate_idx))
        ):
            reason = "course_not_started" if not self.started_course else "does_not_match_expected_gate"
            self._log_throttled(
                f"reject_detection_gate_{self.gate_idx}_{reason}",
                now,
                0.75,
                f"detection_rejected gate={self.gate_idx} reason={reason} det={self._det_summary(raw_detection)}",
            )
            detection = None

        if (
            detection is not None
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

        if self.gate_idx >= self.num_gates:
            return self._return_to_start_or_next_lap(now, pos, lap_after_return=self.route_lap + 1)

        gate = self.gate_estimates[self.gate_idx]
        if gate is None:
            self.stage = "fallback_search"
            return self._search_command(now, pos, self.gate_idx)

        gate = self._correct_stored_gate_from_vision(now, pos, sensor_data, detection, image_shape, gate.copy())
        entry, center, exit_point, pass_dir, pass_yaw = self._gate_waypoints(self.gate_idx, gate)

        if self.stored_stage == "entry":
            self.stage = "stored_entry"
            if np.linalg.norm(pos - entry) < 0.24:
                self._log(f"stored_entry_reached lap={self.route_lap} gate={self.gate_idx}")
                self.stored_stage = "center"
            else:
                return self._limited_setpoint(pos, entry, max_step=0.52, yaw_target=pass_yaw)

        if self.stored_stage == "center":
            self.stage = "stored_center"
            if np.linalg.norm(pos - center) < 0.21:
                self._log(f"stored_center_reached lap={self.route_lap} gate={self.gate_idx}")
                self.stored_stage = "exit"
            else:
                return self._limited_setpoint(pos, center, max_step=0.24, yaw_target=pass_yaw)

        self.stage = "stored_exit"
        progress = float(np.dot(pos[:2] - gate[:2], pass_dir[:2]))
        exit_error = np.linalg.norm(pos - exit_point)
        if progress > 0.76 or (exit_error < 0.24 and progress > 0.58):
            self._log(
                f"stored_gate_done lap={self.route_lap} gate={self.gate_idx} "
                f"progress={progress:.2f} exit_error={exit_error:.2f}"
            )
            self.gate_idx += 1
            self.stored_stage = "entry"
            return self._stored_lap_command(now, pos, yaw, sensor_data, detection, image_shape)

        return self._limited_setpoint(pos, exit_point, max_step=0.38, yaw_target=pass_yaw)

    def _return_to_start_or_next_lap(self, now, pos, lap_after_return):
        self.phase = "return_to_start"
        self.stage = "return"
        segment = self._segment_from_position(pos[:2])
        if segment == 0 and np.linalg.norm(pos[:2] - self.takeoff_pos[:2]) < 0.75:
            self._log(f"own_lap_complete old_lap={self.route_lap} next_lap={lap_after_return}")
            self.route_lap = lap_after_return
            self.gate_idx = 0
            self.stage = "search"
            self.stored_stage = "entry"
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
            self._log(f"search_start gate={gate_idx} expected={self._vec(self._expected_gate_position(gate_idx))}")

        expected_gate = self._expected_gate_position(gate_idx)
        pass_dir = self._pass_direction_for_gate(gate_idx, expected_gate)
        pass_yaw = math.atan2(pass_dir[1], pass_dir[0])
        standoff = expected_gate - 0.75 * pass_dir
        standoff[2] = 1.25

        estimate = self.gate_estimates[gate_idx]
        if estimate is not None and now - self.last_detection_time[gate_idx] < 1.0:
            standoff = estimate - 0.55 * self._pass_direction_for_gate(gate_idx, estimate, pos)
            standoff[2] = np.clip(estimate[2], 0.75, 2.05)
            pass_yaw = self._yaw_to_point(pos, estimate)
            self._log_throttled(
                f"search_using_recent_estimate_{gate_idx}",
                now,
                0.75,
                f"search_using_recent_estimate gate={gate_idx} estimate={self._vec(estimate)} standoff={self._vec(standoff)}",
            )

        dist_to_standoff = np.linalg.norm(pos[:2] - standoff[:2])
        if dist_to_standoff > 0.32:
            self._log_throttled(
                f"search_move_to_standoff_{gate_idx}",
                now,
                1.0,
                f"search_move_to_standoff gate={gate_idx} dist={dist_to_standoff:.2f} target={self._vec(standoff)} yaw={pass_yaw:.3f}",
            )
            return self._limited_setpoint(pos, standoff, max_step=0.50, yaw_target=pass_yaw)

        scan = 0.70 * math.sin(0.55 * (now - self.search_started_at))
        yaw_target = self._wrap_angle(pass_yaw + scan)
        self._log_throttled(
            f"search_scan_{gate_idx}",
            now,
            1.0,
            f"search_scan gate={gate_idx} standoff={self._vec(standoff)} yaw={yaw_target:.3f} scan={scan:.3f}",
        )
        return self._limited_setpoint(pos, standoff, max_step=0.18, yaw_target=yaw_target)

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
        if progress > 0.68 or (exit_error < 0.24 and progress > 0.52):
            self._log(f"pass_done lap=0 gate={gate_idx} progress={progress:.2f} exit_error={exit_error:.2f}")
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
        candidate = mono
        source = "monocular"
        if mono is not None and tri is not None:
            disagreement = np.linalg.norm(mono[:2] - tri[:2])
            if disagreement <= 0.35:
                candidate = 0.65 * mono + 0.35 * tri
                source = "mono_tri_agree"
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

        old = self.gate_estimates[gate_idx]
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
        self.gate_estimates[gate_idx] = estimate
        self.estimate_samples[gate_idx] += 1
        self._log(
            f"estimate_accept gate={gate_idx} source={source} samples={self.estimate_samples[gate_idx]} "
            f"estimate={self._vec(estimate)} candidate={self._vec(candidate)} mono_reason={mono_reason} tri_reason={tri_reason}"
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
                best = {
                    "score": score,
                    "area": float(area),
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "center": (float(cx), float(cy)),
                    "err_x": float((cx - w / 2.0) / (w / 2.0)),
                    "err_y": float((cy - h / 2.0) / (h / 2.0)),
                    "fill": float(fill),
                    "touches_edge": bool(touches_edge),
                    "mask": mask,
                }
        return best

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
        return np.array(
            [
                self.center[0] - self.nominal_radius * math.cos(angle),
                self.center[1] - self.nominal_radius * math.sin(angle),
                1.25,
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

        tangent = self._geometric_pass_direction_for_gate(gate_idx, gate_estimate)
        if gate_estimate is not None and pos is not None:
            to_gate = np.asarray(gate_estimate)[:2] - np.asarray(pos)[:2]
            norm = np.linalg.norm(to_gate)
            if norm > 1e-6:
                approach = np.array([to_gate[0] / norm, to_gate[1] / norm, 0.0])
                if np.dot(approach[:2], tangent[:2]) > 0.35:
                    return approach

        return tangent

    def _learn_pass_direction(self, gate_idx, gate_estimate, pos):
        tangent = self._geometric_pass_direction_for_gate(gate_idx, gate_estimate)
        to_gate = np.asarray(gate_estimate)[:2] - np.asarray(pos)[:2]
        norm = np.linalg.norm(to_gate)
        source = "tangent"
        if norm > 1e-6:
            approach = np.array([to_gate[0] / norm, to_gate[1] / norm, 0.0])
            if np.dot(approach[:2], tangent[:2]) > 0.35:
                direction = approach
                source = "approach"
            else:
                direction = tangent
        else:
            direction = tangent

        old = self.gate_pass_directions[gate_idx]
        if old is not None and np.dot(old[:2], direction[:2]) > 0.0:
            direction = 0.65 * old + 0.35 * direction
            direction /= np.linalg.norm(direction[:2])

        self.gate_pass_directions[gate_idx] = direction.copy()
        self._log(
            f"pass_direction_learned gate={gate_idx} source={source} "
            f"gate={self._vec(gate_estimate)} pos={self._vec(pos)} "
            f"tangent={self._vec(tangent)} direction={self._vec(direction)}"
        )
        return direction

    def _gate_waypoints(self, gate_idx, gate_estimate):
        gate = np.asarray(gate_estimate, dtype=float).copy()
        pass_dir = self._pass_direction_for_gate(gate_idx, gate)
        pass_dir[2] = 0.0
        norm = np.linalg.norm(pass_dir[:2])
        if norm < 1e-9:
            pass_dir = self._geometric_pass_direction_for_gate(gate_idx, gate)
        else:
            pass_dir = pass_dir / norm

        center = gate.copy()
        entry = gate - 0.58 * pass_dir
        exit_point = gate + 0.88 * pass_dir
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
