#!/usr/bin/env python3
import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None

RAW_IMAGE_TYPE = "sensor_msgs/msg/Image"
ULTRALYTICS_INSTALL_HINT = (
    "Install it with `pip3 install -r come_to_user/requirements.txt` "
    "or `pip3 install ultralytics`."
)
START_ARM_SERVICE = "/mars/arm/goto_js"
START_ARM_SERVICE_TYPE = "maurice_msgs/srv/GotoJS"

START_ARM_POSE = [
    0.4141748127291231,
    -0.27765052260730105,
    0.6273981422452273,
    1.179631225884058,
    0.31139809994078516,
    0.923456434307156,
]

def move_arm_to_start_pose(
    service_name: str,
    service_type: str,
    duration_ms: int,
    timeout_sec: float,
) -> None:
    cmd = [
        "ros2",
        "service",
        "call",
        service_name,
        service_type,
        f"{{data: {{data: {START_ARM_POSE}}}, time: {duration_ms}}}",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out waiting for arm start-pose service {service_name} "
            f"after {timeout_sec:.1f} seconds."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Arm start-pose command failed with exit code {exc.returncode}: "
            + " ".join(cmd)
        ) from exc

@dataclass
class CandidateTopic:
    name: str
    type_name: str
    publisher_count: int
    score: int


class PersonFollowerWebServer(Node):
    def __init__(
        self,
        preferred_topics: List[str],
        arm_image_topic: str,
        annotated_topic: str,
        edge_debug_topic: str,
        combined_debug_topic: str,
        cmd_vel_topic: str,
        model_path: str,
        conf_threshold: float,
        imgsz: int,
        device: Optional[str],
        forward_speed: float,
        person_center_tolerance: float,
        turn_gain: float,
        drive_turn_gain: float,
        max_turn_speed: float,
        person_state_timeout: float,
        arm_frame_timeout: float,
        edge_roi_start: float,
        edge_consecutive_frames: int,
        edge_canny_low: int,
        edge_canny_high: int,
        edge_grad_threshold: float,
        edge_fit_tolerance_px: int,
        edge_min_support_fraction: float,
        edge_min_span_fraction: float,
        edge_center_corridor_fraction: float,
        edge_min_corridor_support_fraction: float,
        edge_stop_y_fraction: float,
    ) -> None:
        super().__init__("person_follower_web_server")

        self.preferred_topics = preferred_topics
        self.arm_image_topic = arm_image_topic
        self.annotated_topic = annotated_topic
        self.edge_debug_topic = edge_debug_topic
        self.combined_debug_topic = combined_debug_topic
        self.cmd_vel_topic = cmd_vel_topic

        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

        self.forward_speed = forward_speed
        self.person_center_tolerance = float(np.clip(person_center_tolerance, 0.0, 1.0))
        self.turn_gain = turn_gain
        self.drive_turn_gain = drive_turn_gain
        self.max_turn_speed = max_turn_speed
        self.person_state_timeout = person_state_timeout
        self.arm_frame_timeout = arm_frame_timeout

        self.edge_roi_start = float(np.clip(edge_roi_start, 0.0, 0.95))
        self.edge_consecutive_frames = max(1, edge_consecutive_frames)
        self.edge_canny_low = int(edge_canny_low)
        self.edge_canny_high = int(edge_canny_high)
        self.edge_grad_threshold = float(edge_grad_threshold)
        self.edge_fit_tolerance_px = int(edge_fit_tolerance_px)
        self.edge_min_support_fraction = float(np.clip(edge_min_support_fraction, 0.01, 1.0))
        self.edge_min_span_fraction = float(np.clip(edge_min_span_fraction, 0.01, 1.0))
        self.edge_center_corridor_fraction = float(np.clip(edge_center_corridor_fraction, 0.05, 1.0))
        self.edge_min_corridor_support_fraction = float(
            np.clip(edge_min_corridor_support_fraction, 0.01, 1.0)
        )
        self.edge_stop_y_fraction = float(np.clip(edge_stop_y_fraction, 0.0, 1.0))

        self.bridge = CvBridge()

        self.main_subscription = None

        self.main_cb_group = MutuallyExclusiveCallbackGroup()
        self.arm_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.debug_cb_group = MutuallyExclusiveCallbackGroup()

        self.yolo_publisher = self.create_publisher(
            Image,
            self.annotated_topic,
            qos_profile_sensor_data,
        )
        self.edge_debug_publisher = self.create_publisher(
            Image,
            self.edge_debug_topic,
            qos_profile_sensor_data,
        )
        self.combined_debug_publisher = self.create_publisher(
            Image,
            self.combined_debug_topic,
            qos_profile_sensor_data,
        )
        self.cmd_publisher = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )

        self.arm_subscription = self.create_subscription(
            Image,
            self.arm_image_topic,
            self.arm_image_callback,
            qos_profile_sensor_data,
            callback_group=self.arm_cb_group,
        )

        self.control_timer = self.create_timer(
            0.1,
            self.control_timer_callback,
            callback_group=self.control_cb_group,
        )

        self.combined_debug_timer = self.create_timer(
            0.15,
            self.publish_combined_debug_callback,
            callback_group=self.debug_cb_group,
        )

        if YOLO is None:
            raise RuntimeError(
                "Missing Python package 'ultralytics'. " + ULTRALYTICS_INSTALL_HINT
            )
        self.model = YOLO(model_path)
        self.person_class_ids = self._resolve_person_class_ids()

        self.processing_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.processed_frames = 0
        self.dropped_frames = 0
        self.last_log_time = time.time()
        self.last_control_log_time = 0.0

        self.motion_enabled = False
        self.controller_state = "startup"

        self.person_visible = False
        self.person_error_norm = 0.0
        self.person_bbox_area_fraction = 0.0
        self.person_confidence = 0.0
        self.last_main_state_time = 0.0

        self.arm_frame_received = False
        self.last_arm_frame_time = 0.0
        self.edge_hits_in_row = 0
        self.edge_detected_current = False
        self.edge_latched = False

        self.latest_main_debug_bgr: Optional[np.ndarray] = None
        self.latest_edge_debug_bgr: Optional[np.ndarray] = None

        if not self.person_class_ids:
            self.get_logger().warn(
                "Could not find a 'person' class in the model. "
                "YOLO frames will still publish, but no person tracking will work."
            )

        self.get_logger().info(f"Arm camera topic: {self.arm_image_topic}")
        self.get_logger().info(f"YOLO topic: {self.annotated_topic}")
        self.get_logger().info(f"Edge debug topic: {self.edge_debug_topic}")
        self.get_logger().info(f"Combined debug topic: {self.combined_debug_topic}")
        self.get_logger().info(f"cmd_vel topic: {self.cmd_vel_topic}")

    def _resolve_person_class_ids(self) -> List[int]:
        names = getattr(self.model, "names", {})
        person_ids: List[int] = []

        if isinstance(names, dict):
            for idx, name in names.items():
                if str(name).strip().lower() == "person":
                    person_ids.append(int(idx))
        elif isinstance(names, list):
            for idx, name in enumerate(names):
                if str(name).strip().lower() == "person":
                    person_ids.append(idx)

        return person_ids

    def _score_topic(self, topic_name: str) -> int:
        score = 0
        for idx, preferred in enumerate(self.preferred_topics):
            if topic_name == preferred:
                score += 1000 - idx

        lower = topic_name.lower()
        if "camera" in lower:
            score += 50
        if "image_raw" in lower:
            score += 50
        if "image" in lower:
            score += 20
        if "left" in lower:
            score += 5
        if "front" in lower:
            score += 3
        if "remote" in lower:
            score += 2
        return score

    def find_best_main_image_topic(self) -> Optional[CandidateTopic]:
        best: Optional[CandidateTopic] = None

        for topic_name, type_names in self.get_topic_names_and_types():
            if RAW_IMAGE_TYPE not in type_names:
                continue

            try:
                publisher_infos = self.get_publishers_info_by_topic(topic_name)
                publisher_count = len(publisher_infos)
            except Exception:
                publisher_count = 0

            if publisher_count <= 0:
                continue

            candidate = CandidateTopic(
                name=topic_name,
                type_name=RAW_IMAGE_TYPE,
                publisher_count=publisher_count,
                score=self._score_topic(topic_name),
            )

            if best is None:
                best = candidate
                continue

            if (
                candidate.score > best.score
                or (
                    candidate.score == best.score
                    and candidate.publisher_count > best.publisher_count
                )
                or (
                    candidate.score == best.score
                    and candidate.publisher_count == best.publisher_count
                    and candidate.name < best.name
                )
            ):
                best = candidate

        return best

    def start_main_subscription(self, image_topic: str) -> None:
        if self.main_subscription is not None:
            return

        self.main_subscription = self.create_subscription(
            Image,
            image_topic,
            self.main_image_callback,
            qos_profile_sensor_data,
            callback_group=self.main_cb_group,
        )
        self.get_logger().info(f"Subscribed to main image topic: {image_topic}")

    def enable_motion(self) -> None:
        with self.state_lock:
            self.motion_enabled = True
            self.edge_hits_in_row = 0
            self.edge_latched = False
        self.get_logger().info("Motion enabled.")

    def disable_motion(self) -> None:
        self.motion_enabled = False
        self.publish_stop()
        self.get_logger().info("Motion disabled.")

    def publish_twist(self, linear_x: float, angular_z: float = 0.0) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular_z)
        self.cmd_publisher.publish(msg)

    def publish_stop(self) -> None:
        self.publish_twist(0.0, 0.0)

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def control_timer_callback(self) -> None:
        now = time.time()

        with self.state_lock:
            arm_ready = self.arm_frame_received and (
                (now - self.last_arm_frame_time) <= self.arm_frame_timeout
            )
            person_ready = (now - self.last_main_state_time) <= self.person_state_timeout
            person_visible = self.person_visible
            person_error_norm = self.person_error_norm
            edge_latched = self.edge_latched

        linear_x = 0.0
        angular_z = 0.0

        if not self.motion_enabled:
            state = "HOLD(startup)"
        elif not arm_ready:
            state = "HOLD(waiting for arm camera)"
        elif edge_latched:
            state = "STOP(edge latched)"
        elif not person_ready:
            state = "HOLD(waiting for YOLO)"
        elif not person_visible:
            state = "HOLD(no person)"
        else:
            if abs(person_error_norm) > self.person_center_tolerance:
                angular_z = self.clamp(
                    -self.turn_gain * person_error_norm,
                    -self.max_turn_speed,
                    self.max_turn_speed,
                )
                linear_x = 0.0
                state = f"ALIGN(err={person_error_norm:+.2f})"
            else:
                angular_z = self.clamp(
                    -self.drive_turn_gain * person_error_norm,
                    -self.max_turn_speed,
                    self.max_turn_speed,
                )
                linear_x = self.forward_speed
                state = f"FOLLOW(err={person_error_norm:+.2f})"

        self.publish_twist(linear_x, angular_z)

        with self.state_lock:
            self.controller_state = state

        if now - self.last_control_log_time >= 2.0:
            self.get_logger().info(state)
            self.last_control_log_time = now

    def main_image_callback(self, msg: Image) -> None:
        if not self.processing_lock.acquire(blocking=False):
            self.dropped_frames += 1
            return

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            annotated, target_found, target_error_norm, target_area_fraction, target_conf = self.run_person_detection(
                frame_bgr
            )

            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = msg.header
            self.yolo_publisher.publish(out_msg)

            now = time.time()
            with self.state_lock:
                self.person_visible = target_found
                self.person_error_norm = target_error_norm
                self.person_bbox_area_fraction = target_area_fraction
                self.person_confidence = target_conf
                self.last_main_state_time = now
                self.latest_main_debug_bgr = annotated.copy()

            self.processed_frames += 1
            if now - self.last_log_time >= 5.0:
                self.get_logger().info(
                    f"Processed={self.processed_frames} Dropped={self.dropped_frames}"
                )
                self.last_log_time = now

        except Exception as exc:
            self.get_logger().error(f"Main image processing failed: {exc}")
        finally:
            self.processing_lock.release()

    def run_person_detection(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, bool, float, float, float]:
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        cx_img = w // 2

        cv2.line(annotated, (cx_img, 0), (cx_img, h - 1), (255, 255, 0), 1)
        cv2.putText(
            annotated,
            "frame center",
            (cx_img + 8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        best_area = -1
        best_box = None
        people_count = 0

        if boxes is not None:
            for box in boxes:
                cls_tensor = getattr(box, "cls", None)
                conf_tensor = getattr(box, "conf", None)
                xyxy_tensor = getattr(box, "xyxy", None)

                if cls_tensor is None or conf_tensor is None or xyxy_tensor is None:
                    continue

                class_id = int(cls_tensor[0].item())
                if class_id not in self.person_class_ids:
                    continue

                x1, y1, x2, y2 = [int(v) for v in xyxy_tensor[0].tolist()]
                confidence = float(conf_tensor[0].item())
                area = max(0, x2 - x1) * max(0, y2 - y1)
                people_count += 1

                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2, confidence)

        target_found = best_box is not None
        target_error_norm = 0.0
        target_area_fraction = 0.0
        target_conf = 0.0

        if boxes is not None:
            for box in boxes:
                cls_tensor = getattr(box, "cls", None)
                conf_tensor = getattr(box, "conf", None)
                xyxy_tensor = getattr(box, "xyxy", None)

                if cls_tensor is None or conf_tensor is None or xyxy_tensor is None:
                    continue

                class_id = int(cls_tensor[0].item())
                if class_id not in self.person_class_ids:
                    continue

                x1, y1, x2, y2 = [int(v) for v in xyxy_tensor[0].tolist()]
                confidence = float(conf_tensor[0].item())
                area = max(0, x2 - x1) * max(0, y2 - y1)

                color = (0, 255, 0)
                thickness = 2
                label = f"person {confidence:.2f}"

                if area == best_area and best_box is not None and (x1, y1, x2, y2) == best_box[:4]:
                    color = (0, 0, 255)
                    thickness = 3
                    label = f"TARGET {confidence:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if target_found and best_box is not None:
            x1, y1, x2, y2, target_conf = best_box
            target_cx = int((x1 + x2) / 2)
            target_cy = int((y1 + y2) / 2)
            target_error_norm = float(target_cx - cx_img) / max(1.0, float(w / 2))
            target_area_fraction = float(best_area) / max(1.0, float(w * h))

            cv2.circle(annotated, (target_cx, target_cy), 6, (0, 0, 255), -1)
            cv2.line(annotated, (cx_img, target_cy), (target_cx, target_cy), (0, 0, 255), 2)

            align_text = "CENTERED" if abs(target_error_norm) <= self.person_center_tolerance else "ALIGNING"
            cv2.putText(
                annotated,
                f"{align_text} err={target_error_norm:+.2f} area={target_area_fraction:.3f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                annotated,
                "NO PERSON",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            f"people: {people_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return annotated, target_found, target_error_norm, target_area_fraction, target_conf

    def arm_image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if frame.ndim == 2:
                gray = frame
                debug_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                debug_bgr = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                debug_bgr = bgr.copy()
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Unsupported arm image shape: {frame.shape}")

            detected, debug_bgr, stats = self.detect_edge_in_forward_corridor(gray, debug_bgr)

            debug_msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding="bgr8")
            debug_msg.header = msg.header
            self.edge_debug_publisher.publish(debug_msg)

            now = time.time()
            with self.state_lock:
                self.arm_frame_received = True
                self.last_arm_frame_time = now
                self.edge_detected_current = detected
                edge_stop_armed = self.motion_enabled

                if detected and edge_stop_armed:
                    self.edge_hits_in_row += 1
                else:
                    self.edge_hits_in_row = 0

                if edge_stop_armed and self.edge_hits_in_row >= self.edge_consecutive_frames:
                    self.edge_latched = True

                self.latest_edge_debug_bgr = debug_bgr.copy()
                edge_latched = self.edge_latched
                edge_hits_in_row = self.edge_hits_in_row

            if edge_latched:
                self.publish_stop()

            if detected and edge_hits_in_row == self.edge_consecutive_frames:
                self.get_logger().warn(
                    "Edge latched. "
                    f"support={stats['support_fraction']:.2f} "
                    f"span={stats['x_span_fraction']:.2f} "
                    f"corridor={stats['corridor_support_fraction']:.2f} "
                    f"y_max_corridor={stats['max_curve_y_corridor']:.0f}"
                )

        except Exception as exc:
            self.get_logger().error(f"Arm image processing failed: {exc}")

    def detect_edge_in_forward_corridor(
        self,
        gray: np.ndarray,
        debug_bgr: np.ndarray,
    ) -> Tuple[bool, np.ndarray, Dict[str, float]]:
        h, w = gray.shape
        y0 = int(h * self.edge_roi_start)
        roi = gray[y0:, :]

        stats: Dict[str, float] = {
            "columns_total": 0.0,
            "columns_used": 0.0,
            "support_fraction": 0.0,
            "x_span_fraction": 0.0,
            "corridor_support_fraction": 0.0,
            "max_curve_y_corridor": -1.0,
        }

        if roi.size == 0:
            cv2.putText(
                debug_bgr,
                "ROI empty",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return False, debug_bgr, stats

        corridor_half_w = int(0.5 * self.edge_center_corridor_fraction * w)
        cx = w // 2
        x_left = max(0, cx - corridor_half_w)
        x_right = min(w - 1, cx + corridor_half_w)
        stop_y = int(h * self.edge_stop_y_fraction)

        cv2.rectangle(debug_bgr, (0, y0), (w - 1, h - 1), (255, 255, 0), 2)
        cv2.rectangle(debug_bgr, (x_left, y0), (x_right, h - 1), (0, 165, 255), 2)
        cv2.line(debug_bgr, (0, stop_y), (w - 1, stop_y), (0, 255, 255), 2)

        blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        edges = cv2.Canny(blurred, self.edge_canny_low, self.edge_canny_high)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        debug_bgr[y0:, :] = cv2.addWeighted(debug_bgr[y0:, :], 0.8, edges_bgr, 0.2, 0.0)

        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_y = np.abs(grad_y)
        grad_y = cv2.GaussianBlur(grad_y, (5, 5), 0)

        step = max(2, w // 160)
        sampled_columns = list(range(0, w, step))
        stats["columns_total"] = float(len(sampled_columns))

        xs: List[float] = []
        ys: List[float] = []

        corridor_columns_total = 0
        for x in sampled_columns:
            if x_left <= x <= x_right:
                corridor_columns_total += 1

            col = grad_y[:, x]
            y_local = int(np.argmax(col))
            strength = float(col[y_local])

            if strength >= self.edge_grad_threshold:
                xs.append(float(x))
                ys.append(float(y0 + y_local))

        stats["columns_used"] = float(len(xs))

        if len(sampled_columns) == 0:
            return False, debug_bgr, stats

        if len(xs) < max(12, int(0.10 * len(sampled_columns))):
            for x, y in zip(xs, ys):
                cv2.circle(debug_bgr, (int(x), int(y)), 2, (0, 165, 255), -1)

            cv2.putText(
                debug_bgr,
                f"few points: {len(xs)}/{len(sampled_columns)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_bgr,
                "EDGE: NO",
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            return False, debug_bgr, stats

        xs_np = np.array(xs, dtype=np.float32)
        ys_np = np.array(ys, dtype=np.float32)

        degree = 2 if len(xs_np) >= 6 else 1
        coeff = np.polyfit(xs_np, ys_np, degree)
        y_fit = np.polyval(coeff, xs_np)

        residuals = np.abs(ys_np - y_fit)
        inliers = residuals <= self.edge_fit_tolerance_px

        if np.count_nonzero(inliers) >= max(8, int(0.05 * len(sampled_columns))):
            coeff = np.polyfit(xs_np[inliers], ys_np[inliers], degree)
            y_fit = np.polyval(coeff, xs_np)
            residuals = np.abs(ys_np - y_fit)
            inliers = residuals <= self.edge_fit_tolerance_px

        support_fraction = float(np.count_nonzero(inliers)) / float(len(sampled_columns))
        stats["support_fraction"] = support_fraction

        if np.any(inliers):
            x_in = xs_np[inliers]
            x_span_fraction = float((x_in.max() - x_in.min()) / max(1.0, float(w)))
        else:
            x_span_fraction = 0.0
        stats["x_span_fraction"] = x_span_fraction

        corridor_mask = (xs_np >= x_left) & (xs_np <= x_right)
        corridor_inliers = inliers & corridor_mask
        corridor_support_fraction = (
            float(np.count_nonzero(corridor_inliers)) / float(max(1, corridor_columns_total))
        )
        stats["corridor_support_fraction"] = corridor_support_fraction

        for x, y, ok in zip(xs_np, ys_np, inliers):
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.circle(debug_bgr, (int(x), int(y)), 2, color, -1)

        curve_x = np.arange(0, w, 4, dtype=np.float32)
        curve_y = np.polyval(coeff, curve_x)

        curve_pts = []
        corridor_curve_y = []
        for x, y in zip(curve_x, curve_y):
            yi = int(round(float(y)))
            if 0 <= yi < h:
                curve_pts.append([int(x), yi])
                if x_left <= int(x) <= x_right:
                    corridor_curve_y.append(yi)

        if len(curve_pts) >= 2:
            curve_pts_np = np.array(curve_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_bgr, [curve_pts_np], False, (255, 0, 255), 2)

        max_curve_y_corridor = max(corridor_curve_y) if corridor_curve_y else -1
        stats["max_curve_y_corridor"] = float(max_curve_y_corridor)

        detected = (
            support_fraction >= self.edge_min_support_fraction
            and x_span_fraction >= self.edge_min_span_fraction
            and corridor_support_fraction >= self.edge_min_corridor_support_fraction
            and max_curve_y_corridor >= stop_y
        )

        text1 = (
            f"support={support_fraction:.2f} span={x_span_fraction:.2f} "
            f"corridor={corridor_support_fraction:.2f}"
        )
        text2 = f"y_max_corridor={max_curve_y_corridor:.0f} stop_y={stop_y}"
        cv2.putText(
            debug_bgr,
            text1,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if detected else (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_bgr,
            text2,
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if detected else (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_bgr,
            "EDGE: YES" if detected else "EDGE: NO",
            (15, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255) if detected else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return detected, debug_bgr, stats

    def make_placeholder(self, text: str, width: int = 640, height: int = 480) -> np.ndarray:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        cv2.putText(
            img,
            text,
            (30, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        return img

    def resize_to_height(self, img: np.ndarray, target_height: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == target_height:
            return img
        scale = target_height / float(h)
        target_width = max(1, int(round(w * scale)))
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    def publish_combined_debug_callback(self) -> None:
        with self.state_lock:
            main_img = None if self.latest_main_debug_bgr is None else self.latest_main_debug_bgr.copy()
            edge_img = None if self.latest_edge_debug_bgr is None else self.latest_edge_debug_bgr.copy()
            controller_state = self.controller_state
            person_visible = self.person_visible
            person_error_norm = self.person_error_norm
            edge_latched = self.edge_latched

        if main_img is None:
            main_img = self.make_placeholder("Waiting for main camera / YOLO")
        if edge_img is None:
            edge_img = self.make_placeholder("Waiting for arm camera / edge detector")

        target_height = max(main_img.shape[0], edge_img.shape[0])
        main_resized = self.resize_to_height(main_img, target_height)
        edge_resized = self.resize_to_height(edge_img, target_height)

        cv2.putText(
            main_resized,
            "YOLO / person tracking",
            (15, target_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            edge_resized,
            "Arm camera / edge stop",
            (15, target_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        stacked = np.hstack([main_resized, edge_resized])

        banner_h = 50
        banner = np.zeros((banner_h, stacked.shape[1], 3), dtype=np.uint8)
        banner[:] = (20, 20, 20)

        summary = (
            f"state={controller_state} | "
            f"person_visible={person_visible} | "
            f"person_err={person_error_norm:+.2f} | "
            f"edge_latched={edge_latched}"
        )
        cv2.putText(
            banner,
            summary,
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combined = np.vstack([banner, stacked])

        msg = self.bridge.cv2_to_imgmsg(combined, encoding="bgr8")
        self.combined_debug_publisher.publish(msg)


def resolve_robot_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def resolve_robot_ips(override_ip: Optional[str] = None) -> List[str]:
    if override_ip:
        return [override_ip]

    candidates: List[str] = [resolve_robot_ip()]

    try:
        hostname_ips = subprocess.run(
            ["hostname", "-I"],
            check=False,
            capture_output=True,
            text=True,
        )
        if hostname_ips.returncode == 0:
            candidates.extend(hostname_ips.stdout.split())
    except Exception:
        pass

    try:
        candidates.extend(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass

    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or ":" in candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)

    non_loopback = [ip for ip in unique if not ip.startswith("127.")]
    loopback = [ip for ip in unique if ip.startswith("127.")]
    return non_loopback + loopback if non_loopback else loopback or ["127.0.0.1"]


def build_web_video_urls(host: str, port: int, topic: str) -> Tuple[str, str, str]:
    encoded_topic = quote(topic, safe="")
    base_url = f"http://{host}:{port}"
    viewer_url = (
        f"{base_url}/stream_viewer"
        f"?topic={encoded_topic}&qos_profile=sensor_data"
    )
    stream_url = (
        f"{base_url}/stream"
        f"?topic={encoded_topic}&qos_profile=sensor_data"
    )
    return base_url, viewer_url, stream_url


def print_web_video_urls(hosts: List[str], port: int, topic: str) -> None:
    print("\nOpen this on your laptop:")
    for host in hosts:
        base_url, viewer_url, stream_url = build_web_video_urls(host, port, topic)
        print(f"  Server root:         {base_url}/")
        print(f"  Combined debug view: {viewer_url}")
        print(f"  Combined raw stream: {stream_url}")
    if hosts == ["127.0.0.1"]:
        print("  Only loopback was detected. Pass --robot-ip ROBOT_LAN_IP if opening from another machine.")


def check_ros_package_available(package_name: str, install_hint: str) -> None:
    try:
        subprocess.run(
            ["ros2", "pkg", "prefix", package_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"ERROR: ROS package '{package_name}' is not available.\n{install_hint}",
            file=sys.stderr,
        )
        sys.exit(2)


def check_ultralytics_available() -> None:
    if YOLO is not None:
        return

    print(
        "ERROR: Python package 'ultralytics' is not available.\n"
        + ULTRALYTICS_INSTALL_HINT,
        file=sys.stderr,
    )
    sys.exit(2)


def launch_web_video_server(address: str, port: int, verbose: bool) -> subprocess.Popen:
    cmd = [
        "ros2",
        "run",
        "web_video_server",
        "web_video_server",
        "--ros-args",
        "-p",
        f"address:={address}",
        "-p",
        f"port:={port}",
    ]

    if verbose:
        cmd += ["-p", "verbose:=true"]

    print("Launching web_video_server:")
    print("  " + " ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def stop_process_group(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the closest person in the main camera, align to center, follow forward, "
            "and stop if the arm camera detects an edge entering the forward corridor."
        )
    )
    parser.add_argument(
        "--preferred",
        nargs="*",
        default=["/mars/main_camera/remote/left/image_raw"],
        help="Preferred main raw image topics in priority order.",
    )
    parser.add_argument(
        "--arm-image-topic",
        default="/mars/arm/image_raw",
        help="Arm camera topic used for edge detection.",
    )
    parser.add_argument(
        "--skip-arm-start",
        action="store_true",
        help="Skip moving the arm to the start pose before person following.",
    )
    parser.add_argument(
        "--arm-start-service",
        default=START_ARM_SERVICE,
        help="Arm service used to move to the start pose.",
    )
    parser.add_argument(
        "--arm-start-service-type",
        default=START_ARM_SERVICE_TYPE,
        help="ROS service type for the arm start-pose service.",
    )
    parser.add_argument(
        "--arm-start-duration-ms",
        type=int,
        default=3,
        help="Duration value sent to the arm start-pose service request.",
    )
    parser.add_argument(
        "--arm-start-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for the arm start-pose service call before failing.",
    )
    parser.add_argument(
        "--annotated-topic",
        default="/people_annotated/image_raw",
        help="ROS topic for YOLO annotated main-camera frames.",
    )
    parser.add_argument(
        "--edge-debug-topic",
        default="/arm_edge_debug/image_raw",
        help="ROS topic for arm-camera edge debug frames.",
    )
    parser.add_argument(
        "--combined-debug-topic",
        default="/combined_debug/image_raw",
        help="ROS topic for the combined side-by-side debug view.",
    )
    parser.add_argument(
        "--cmd-vel-topic",
        default="/cmd_vel",
        help="Velocity command topic.",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Ultralytics model path or model name.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu, cuda:0, mps.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="web_video_server HTTP port.",
    )
    parser.add_argument(
        "--address",
        default="0.0.0.0",
        help="web_video_server bind address.",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for an active main image topic.",
    )
    parser.add_argument(
        "--robot-ip",
        default=None,
        help="Override the IP shown in the printed browser URLs.",
    )
    parser.add_argument(
        "--verbose-web",
        action="store_true",
        help="Enable verbose web_video_server logs.",
    )

    parser.add_argument(
        "--forward-speed",
        type=float,
        default=0.05,
        help="Forward speed in m/s once the person is centered.",
    )
    parser.add_argument(
        "--person-center-tolerance",
        type=float,
        default=0.10,
        help="Normalized horizontal error tolerance before switching from rotate-in-place to forward motion.",
    )
    parser.add_argument(
        "--turn-gain",
        type=float,
        default=0.8,
        help="Angular gain while rotating in place to center the person.",
    )
    parser.add_argument(
        "--drive-turn-gain",
        type=float,
        default=0.4,
        help="Small steering gain while driving forward.",
    )
    parser.add_argument(
        "--max-turn-speed",
        type=float,
        default=0.8,
        help="Maximum absolute angular.z command.",
    )
    parser.add_argument(
        "--person-state-timeout",
        type=float,
        default=1.0,
        help="How long YOLO target state remains valid.",
    )
    parser.add_argument(
        "--arm-frame-timeout",
        type=float,
        default=1.0,
        help="Stop if the arm camera stops updating for this many seconds.",
    )

    parser.add_argument(
        "--edge-roi-start",
        type=float,
        default=0.50,
        help="Start of arm-camera ROI as a fraction of image height.",
    )
    parser.add_argument(
        "--edge-consecutive-frames",
        type=int,
        default=30,
        help="Number of consecutive edge detections required to latch stop; 20 frames is about 2 seconds at 10 FPS.",
    )
    parser.add_argument(
        "--edge-canny-low",
        type=int,
        default=45,
        help="Low Canny threshold for arm debug visualization.",
    )
    parser.add_argument(
        "--edge-canny-high",
        type=int,
        default=140,
        help="High Canny threshold for arm debug visualization.",
    )
    parser.add_argument(
        "--edge-grad-threshold",
        type=float,
        default=15.0,
        help="Minimum vertical gradient strength for a column to count toward the boundary.",
    )
    parser.add_argument(
        "--edge-fit-tolerance-px",
        type=int,
        default=28,
        help="Allowed deviation from the fitted boundary curve.",
    )
    parser.add_argument(
        "--edge-min-support-fraction",
        type=float,
        default=0.12,
        help="Minimum overall inlier support fraction across sampled columns.",
    )
    parser.add_argument(
        "--edge-min-span-fraction",
        type=float,
        default=0.18,
        help="Minimum horizontal span fraction of the fitted boundary support.",
    )
    parser.add_argument(
        "--edge-center-corridor-fraction",
        type=float,
        default=0.55,
        help="Width fraction of the forward corridor centered in the arm image.",
    )
    parser.add_argument(
        "--edge-min-corridor-support-fraction",
        type=float,
        default=0.12,
        help="Minimum support fraction inside the forward corridor.",
    )
    parser.add_argument(
        "--edge-stop-y-fraction",
        type=float,
        default=0.68,
        help="Stop once the fitted boundary reaches this height fraction in the forward corridor.",
    )

    args = parser.parse_args()

    check_ultralytics_available()
    check_ros_package_available(
        "web_video_server",
        "Install web_video_server for your ROS 2 distro, then rerun this script.",
    )

    if args.skip_arm_start:
        print("Skipping arm start pose.")
    else:
        print(f"Moving arm to start pose with {args.arm_start_service}...")
        try:
            move_arm_to_start_pose(
                service_name=args.arm_start_service,
                service_type=args.arm_start_service_type,
                duration_ms=args.arm_start_duration_ms,
                timeout_sec=args.arm_start_timeout,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
        print("Arm is in start pose.")

    rclpy.init()
    node = PersonFollowerWebServer(
        preferred_topics=args.preferred,
        arm_image_topic=args.arm_image_topic,
        annotated_topic=args.annotated_topic,
        edge_debug_topic=args.edge_debug_topic,
        combined_debug_topic=args.combined_debug_topic,
        cmd_vel_topic=args.cmd_vel_topic,
        model_path=args.model,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        forward_speed=args.forward_speed,
        person_center_tolerance=args.person_center_tolerance,
        turn_gain=args.turn_gain,
        drive_turn_gain=args.drive_turn_gain,
        max_turn_speed=args.max_turn_speed,
        person_state_timeout=args.person_state_timeout,
        arm_frame_timeout=args.arm_frame_timeout,
        edge_roi_start=args.edge_roi_start,
        edge_consecutive_frames=args.edge_consecutive_frames,
        edge_canny_low=args.edge_canny_low,
        edge_canny_high=args.edge_canny_high,
        edge_grad_threshold=args.edge_grad_threshold,
        edge_fit_tolerance_px=args.edge_fit_tolerance_px,
        edge_min_support_fraction=args.edge_min_support_fraction,
        edge_min_span_fraction=args.edge_min_span_fraction,
        edge_center_corridor_fraction=args.edge_center_corridor_fraction,
        edge_min_corridor_support_fraction=args.edge_min_corridor_support_fraction,
        edge_stop_y_fraction=args.edge_stop_y_fraction,
    )

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    proc = launch_web_video_server(args.address, args.port, args.verbose_web)
    time.sleep(1.0)

    if proc.poll() is not None:
        node.publish_stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("ERROR: web_video_server exited immediately.", file=sys.stderr)
        sys.exit(proc.returncode or 1)

    print_web_video_urls(
        resolve_robot_ips(args.robot_ip),
        args.port,
        args.combined_debug_topic,
    )
    print("\nThe debug page can load before camera frames are ready; it will show placeholders until topics arrive.")

    selected_topic: Optional[CandidateTopic] = None
    deadline = time.time() + args.discovery_timeout

    print("Inspecting ROS graph for active main raw image topics...")
    while time.time() < deadline:
        executor.spin_once(timeout_sec=0.25)
        selected_topic = node.find_best_main_image_topic()
        if selected_topic is not None:
            break
        time.sleep(0.1)

    if selected_topic is None:
        node.publish_stop()
        stop_process_group(proc)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print(
            "ERROR: No active main sensor_msgs/msg/Image topic found.\n"
            "Try:\n"
            "  ros2 topic list -t | grep Image\n"
            "  ros2 topic info /mars/main_camera/remote/left/image_raw\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Selected main raw topic: {selected_topic.name} "
        f"(publishers={selected_topic.publisher_count}, score={selected_topic.score})"
    )
    print(f"Arm image topic:        {args.arm_image_topic}")
    print(f"YOLO topic:             {args.annotated_topic}")
    print(f"Edge debug topic:       {args.edge_debug_topic}")
    print(f"Combined debug topic:   {args.combined_debug_topic}")
    print(f"cmd_vel topic:          {args.cmd_vel_topic}")

    node.start_main_subscription(selected_topic.name)
    print("\nBehavior:")
    print("  1) Detect closest person = biggest person bounding box")
    print("  2) Rotate in place until centered")
    print("  3) Drive forward toward person")
    print("  4) Stop and latch stop if the arm-camera boundary enters the forward corridor")
    print("\nKeep this process running. Press Ctrl-C to stop.\n")

    node.enable_motion()

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        node.disable_motion()
        stop_process_group(proc)
        try:
            executor.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
