#!/usr/bin/env python3
"""
Arm Controller: sends motion commands to robotic arm.
Supports PiPER (AgileX) and simulation mode.
"""
import numpy as np
import time


class ArmController:
    def __init__(self, arm_type="piper", simulation=False,
                 max_speed=0.3, smoothing_alpha=0.3):
        """
        arm_type: 'piper' or generic
        simulation: if True, no hardware commands
        max_speed: max end-effector speed in m/s
        smoothing_alpha: low-pass filter coefficient (0-1), lower = smoother
        """
        self.arm_type = arm_type
        self.simulation = simulation
        self.max_speed = max_speed
        self.alpha = smoothing_alpha
        self.current_pose = None
        self.last_command_time = None

    def initialize(self):
        """Connect to arm and move to home position."""
        if self.simulation:
            self.current_pose = np.array([0.0, 0.2, 0.3, 0, 0, 0])
            print("[SIM] Arm initialized at home")
            return

        # PiPER SDK initialization
        try:
            import piper_sdk
            self.arm = piper_sdk.CArmPort()
            self.arm.CreateArm()
            self.arm.ConnectToArm("192.168.7.38")
            time.sleep(1)
            self.arm.ArmMotionCtrl(1)
            self.arm.ArmMotionMode(0)
            self.arm.ArmGripperCtrl(1)
            print("PiPER arm connected")
        except ImportError:
            print("PiPER SDK not found, falling back to simulation")
            self.simulation = True
            self.current_pose = np.array([0.0, 0.2, 0.3, 0, 0, 0])

    def move_to_pose(self, target_pose, timeout=5.0):
        """
        Move end-effector to target pose with speed limiting and smoothing.
        target_pose: [x, y, z, roll, pitch, yaw]
        """
        target_pose = np.array(target_pose)

        if self.current_pose is None:
            self.current_pose = target_pose
            return

        # Speed limiting: cap displacement per command
        now = time.time()
        if self.last_command_time is None:
            dt = 0.033
        else:
            dt = now - self.last_command_time
        self.last_command_time = now

        max_displacement = self.max_speed * dt
        displacement = target_pose[:3] - self.current_pose[:3]
        dist = np.linalg.norm(displacement)

        if dist > max_displacement:
            target_pose[:3] = self.current_pose[:3] + displacement / dist * max_displacement

        # Exponential low-pass filter for smooth motion
        smoothed = self.current_pose + self.alpha * (target_pose - self.current_pose)

        if self.simulation:
            self.current_pose = smoothed
            return smoothed

        # Send to PiPER arm
        self._send_piper_command(smoothed)
        self.current_pose = smoothed
        return smoothed

    def _send_piper_command(self, pose):
        """Convert pose to PiPER joint commands."""
        x, y, z, roll, pitch, yaw = pose

        # PiPER uses position in mm, angles in radians
        msg = {
            "x": x * 1000,
            "y": y * 1000,
            "z": z * 1000,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
        # In real deployment:
        # self.arm.MoveToPoseCartesian(msg)
        pass

    def emergency_stop(self):
        """Emergency stop."""
        if not self.simulation:
            self.arm.ArmMotionCtrl(0)
        print("EMERGENCY STOP")
