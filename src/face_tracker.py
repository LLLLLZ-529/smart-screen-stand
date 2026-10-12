#!/usr/bin/env python3
"""
Face Tracker: MediaPipe Face Mesh based head pose estimation.
Outputs 3D face position and orientation in camera frame.
"""
import cv2
import numpy as np
import mediapipe as mp


class FaceTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Key landmark indices for head pose
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_EYE = 33
        self.RIGHT_EYE = 263
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291
        # Iris centers
        self.LEFT_IRIS = 468
        self.RIGHT_IRIS = 473

    def detect(self, frame):
        """
        Process one frame, return face landmarks and head pose.
        Returns None if no face detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0]
        # Convert normalized landmarks to pixel coordinates
        points = np.array([[p.x * w, p.y * h, p.z] for p in lm.landmark])

        # 2D image points for head pose estimation
        image_points = np.array([
            points[self.NOSE_TIP][:2],
            points[self.CHIN][:2],
            points[self.LEFT_EYE][:2],
            points[self.RIGHT_EYE][:2],
            points[self.LEFT_MOUTH][:2],
            points[self.RIGHT_MOUTH][:2],
        ], dtype=np.float64)

        # 3D model points (generic head model)
        model_points = np.array([
            [0.0, 0.0, 0.0],          # nose tip
            [0.0, -63.6, -12.5],       # chin
            [-43.3, 32.7, -26.0],      # left eye
            [43.3, 32.7, -26.0],       # right eye
            [-28.9, -28.9, -24.1],     # left mouth
            [28.9, -28.9, -24.1],      # right mouth
        ])

        # Camera intrinsics (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP for head pose
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        euler = self._rotation_matrix_to_euler(rotation_matrix)

        # Iris centers for depth estimation
        left_iris = points[self.LEFT_IRIS][:2]
        right_iris = points[self.RIGHT_IRIS][:2]
        iris_distance_px = np.linalg.norm(left_iris - right_iris)

        return {
            "landmarks": points,
            "nose_2d": points[self.NOSE_TIP][:2],
            "head_position": tvec.flatten(),     # [x, y, z] in camera frame
            "head_rotation": euler,             # [pitch, yaw, roll] in degrees
            "iris_distance_px": iris_distance_px,
            "face_center": np.mean([
                points[self.LEFT_EYE][:2],
                points[self.RIGHT_EYE][:2],
            ], axis=0),
        }

    @staticmethod
    def _rotation_matrix_to_euler(R):
        """Convert rotation matrix to Euler angles (degrees)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees([x, y, z])
