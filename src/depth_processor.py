#!/usr/bin/env python3
"""
Depth Processor: RGB-D depth data handling and 3D coordinate reconstruction.
Supports Orbbec / RealSense cameras.
"""
import numpy as np


class DepthProcessor:
    def __init__(self, camera_intrinsics=None):
        """
        camera_intrinsics: dict with fx, fy, cx, cy
        """
        self.intrinsics = camera_intrinsics or {
            "fx": 616.0, "fy": 616.0,
            "cx": 320.0, "cy": 240.0,
        }

    def pixel_to_3d(self, x, y, depth_m):
        """
        Convert pixel coordinates + depth to 3D point in camera frame.
        x, y: pixel coordinates
        depth_m: depth in meters
        Returns: [X, Y, Z] in meters (camera optical frame)
        """
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]

        Z = depth_m
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        return np.array([X, Y, Z])

    def get_face_3d_position(self, face_result, depth_frame):
        """
        Extract smoothed 3D position of face center using depth data.
        Uses a small region around nose tip for robust depth.
        """
        nose_x, nose_y = face_result["nose_2d"].astype(int)
        h, w = depth_frame.shape[:2]

        # Sample depth in a small region for robustness
        region_size = 5
        depths = []
        for dy in range(-region_size, region_size + 1):
            for dx in range(-region_size, region_size + 1):
                px, py = nose_x + dx, nose_y + dy
                if 0 <= px < w and 0 <= py < h:
                    d = depth_frame[py, px]
                    if d > 0:
                        depths.append(d)

        if not depths:
            return None

        # Use median depth to filter outliers
        depth_m = np.median(depths) * 0.001  # mm → m (adjust for camera)

        # Reconstruct 3D point
        point_3d = self.pixel_to_3d(nose_x, nose_y, depth_m)
        return point_3d

    def estimate_distance_from_iris(self, iris_distance_px, real_iris_distance_m=0.063):
        """
        Fallback: estimate distance from iris pixel distance.
        Average human inter-pupillary distance ~63mm.
        """
        fx = self.intrinsics["fx"]
        if iris_distance_px < 1:
            return None
        return fx * real_iris_distance_m / iris_distance_px
