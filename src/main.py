#!/usr/bin/env python3
"""
Smart Screen Stand: Main pipeline.
Camera → face detection → 3D position → coordinate transform → arm control.

Usage:
    python src/main.py                          # camera + arm
    python src/main.py --sim                    # simulation only
    python src/main.py --video test.mp4         # video file
    python src/main.py --no-arm                 # vision only, no arm
"""
import argparse
import sys
import time
import cv2
import numpy as np

from face_tracker import FaceTracker
from depth_processor import DepthProcessor
from coordinate_transform import CoordinateTransform
from arm_controller import ArmController


def get_camera(camera_type="rgb", video_path=None):
    """Initialize camera capture."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def get_depth_frame(cap, camera_type="rgb"):
    """
    Get depth frame from RGB-D camera.
    For RGB-only webcam, returns synthetic depth from iris estimation.
    """
    # Real RGB-D (Orbbec/RealSense):
    # frames = pipeline.wait_for_frames()
    # depth = np.asanyarray(frames.get_depth_frame().get_data())
    return None


def main():
    parser = argparse.ArgumentParser(description="Smart Screen Stand")
    parser.add_argument("--sim", action="store_true", help="simulation mode")
    parser.add_argument("--video", type=str, help="video file input")
    parser.add_argument("--no-arm", action="store_true", help="vision only")
    parser.add_argument("--screen-distance", type=float, default=0.55,
                        help="target eye-to-screen distance (m)")
    args = parser.parse_args()

    # Initialize modules
    face_tracker = FaceTracker()
    depth_processor = DepthProcessor()
    coord_transform = CoordinateTransform()
    arm = ArmController(simulation=args.sim or args.no_arm)
    arm.initialize()

    cap = get_camera(video_path=args.video)

    # FPS tracking
    prev_time = time.time()
    fps_list = []
    frame_count = 0
    face_detected_count = 0

    print("Pipeline running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Face detection + head pose
        face_result = face_tracker.detect(frame)

        if face_result is None:
            cv2.putText(frame, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frame_count += 1
            face_detected_count += 1

            # 2. Get 3D face position
            depth_frame = get_depth_frame(cap)
            if depth_frame is not None:
                face_3d_camera = depth_processor.get_face_3d_position(
                    face_result, depth_frame)
            else:
                # Fallback: estimate from iris distance
                dist = depth_processor.estimate_distance_from_iris(
                    face_result["iris_distance_px"])
                if dist is not None:
                    face_3d_camera = depth_processor.pixel_to_3d(
                        face_result["face_center"][0],
                        face_result["face_center"][1],
                        dist
                    )
                else:
                    face_3d_camera = None

            if face_3d_camera is not None:
                # 3. Transform to robot base frame
                face_3d_base = coord_transform.camera_to_robot_base(face_3d_camera)

                # 4. Compute target screen pose
                target_pose = coord_transform.compute_target_screen_pose(
                    face_3d_base,
                    face_result["head_rotation"],
                    screen_distance=args.screen_distance,
                )

                # 5. Move arm
                arm.move_to_pose(target_pose)

            # Draw visualization
            cx, cy = face_result["face_center"].astype(int)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            pitch, yaw, roll = face_result["head_rotation"]
            cv2.putText(frame, f"Yaw:{yaw:.0f} Pitch:{pitch:.0f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        fps_list.append(fps)
        prev_time = now
        cv2.putText(frame, f"FPS:{fps:.0f}", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Smart Screen Stand", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    if frame_count > 0:
        avg_fps = np.mean(fps_list)
        detection_rate = face_detected_count / max(len(fps_list), 1)
        print(f"\n=== Stats ===")
        print(f"Frames: {len(fps_list)}, Avg FPS: {avg_fps:.1f}")
        print(f"Face detection rate: {detection_rate:.1%}")


if __name__ == "__main__":
    main()
