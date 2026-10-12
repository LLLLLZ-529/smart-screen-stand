#!/usr/bin/env python3
"""
Coordinate Transform: camera frame → robot base frame → arm target pose.
"""
import numpy as np


class CoordinateTransform:
    def __init__(self, camera_to_base=None):
        """
        camera_to_base: 4x4 homogeneous transform from camera to robot base.
        Must be calibrated per installation.
        """
        if camera_to_base is None:
            # Default: camera mounted above arm, looking forward
            # Translation in meters, rotation from camera optical frame to base
            self.T_camera_base = np.eye(4)
            self.T_camera_base[:3, 3] = [0.0, 0.25, 0.40]  # camera 25cm above, 40cm forward
        else:
            self.T_camera_base = np.array(camera_to_base)

    def camera_to_robot_base(self, point_camera):
        """Transform a 3D point from camera frame to robot base frame."""
        p = np.append(point_camera, 1.0)
        return (self.T_camera_base @ p)[:3]

    def compute_target_screen_pose(self, face_3d_base, head_rotation,
                                   screen_distance=0.55, screen_offset=None):
        """
        Compute target end-effector pose so screen faces the user.

        face_3d_base: [x, y, z] face position in robot base frame
        head_rotation: [pitch, yaw, roll] in degrees
        screen_distance: desired eye-to-screen distance in meters
        Returns: [x, y, z, roll, pitch, yaw] target pose
        """
        fx, fy, fz = face_3d_base
        pitch, yaw, roll = head_rotation

        # Target screen position: offset from face toward base
        yaw_rad = np.radians(yaw)
        # Screen sits in front of face, facing the face
        tx = fx - screen_distance * np.sin(yaw_rad)
        tz = fz - screen_distance * np.cos(yaw_rad)
        ty = fy  # keep height aligned with face

        # Screen orientation: face the user (opposite of face normal)
        target_roll = np.radians(roll)
        target_pitch = np.radians(-pitch)
        target_yaw = np.radians(yaw + 180)

        return np.array([tx, ty, tz, target_roll, target_pitch, target_yaw])

    @staticmethod
    def calibrate_transform(camera_points, base_points):
        """
        Estimate camera-to-base transform using paired calibration points.
        Simple rigid transform (Kabsch-like).
        """
        camera_points = np.array(camera_points)
        base_points = np.array(base_points)

        centroid_c = camera_points.mean(axis=0)
        centroid_b = base_points.mean(axis=0)

        H = (camera_points - centroid_c).T @ (base_points - centroid_b)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        T = centroid_b - R @ centroid_c

        T_mat = np.eye(4)
        T_mat[:3, :3] = R
        T_mat[:3, 3] = T
        return T_mat
