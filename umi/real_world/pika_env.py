import pathlib
import sys
import argparse
import numpy as np
import time
import shutil
import math
import cv2

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from multiprocessing.managers import SharedMemoryManager
from umi.real_world.realman_controller import RealmanInterpolationController
from umi.real_world.pika_controller import (
    DEFAULT_GRIPPER_SERIAL_PORT,
    PikaController,
)
from umi.real_world.pika_camera import (
    DEFAULT_FISHEYE_DEVICE,
    DEFAULT_FPS,
    DEFAULT_REALSENSE_SERIAL,
    DEFAULT_RESOLUTION,
    PikaCamera,
)
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from umi.common.pose_util import pose_to_mat, mat_to_pose


def _rot_y(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)


def _rot_z(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _make_transform(rot, trans=None):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    if trans is not None:
        out[:3, 3] = np.asarray(trans, dtype=np.float64)
    return out


def _euler_to_mat(euler):
    rx, ry, rz = euler
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ], dtype=np.float64)


def _mat_to_euler(rot):
    sy = np.clip(-rot[2, 0], -1.0, 1.0)
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-9:
        rx = math.atan2(rot[2, 1], rot[2, 2])
        rz = math.atan2(rot[1, 0], rot[0, 0])
    else:
        rx = 0.0
        rz = math.atan2(-rot[0, 1], rot[1, 1])
    return np.array([rx, ry, rz], dtype=np.float64)


def _realman_euler_pose_to_mat(pose):
    pose = np.asarray(pose, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _euler_to_mat(pose[3:])
    out[:3, 3] = pose[:3]
    return out


def _mat_to_realman_euler_pose(mat):
    pose = np.zeros((6,), dtype=np.float64)
    pose[:3] = mat[:3, 3]
    pose[3:] = _mat_to_euler(mat[:3, :3])
    return pose


# Fixed frame transform from Realman TCP to the Pika gripper frame.
# In right-handed convention, y=-90 maps Realman x to +z and z=+90 maps x to +y.
T_REALMAN_TCP_PIKA_GRIPPER = _make_transform(
    _rot_z(math.pi / 2.0) @ _rot_y(-math.pi / 2.0)
)
if not np.allclose(T_REALMAN_TCP_PIKA_GRIPPER[:3, 0], [0.0, 0.0, 1.0], atol=1e-9):
    raise RuntimeError("Pika gripper x-axis must align with Realman TCP +z-axis.")
T_PIKA_GRIPPER_REALMAN_TCP = np.linalg.inv(T_REALMAN_TCP_PIKA_GRIPPER)


def _realman_tcp_pose_to_pika_gripper_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 1:
        return mat_to_pose(_realman_euler_pose_to_mat(pose) @ T_REALMAN_TCP_PIKA_GRIPPER)
    if pose.ndim != 2 or pose.shape[-1] != 6:
        raise RuntimeError(f"Expected Realman TCP pose shape (..., 6), got {pose.shape}.")
    return np.stack([
        mat_to_pose(_realman_euler_pose_to_mat(x) @ T_REALMAN_TCP_PIKA_GRIPPER)
        for x in pose
    ], axis=0)


def _pika_gripper_pose_to_realman_tcp_pose(pose):
    return _mat_to_realman_euler_pose(pose_to_mat(pose) @ T_PIKA_GRIPPER_REALMAN_TCP)


def _dedupe_time_series(t, x, name):
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x)
    if len(t) != len(x):
        raise RuntimeError(f"{name} timestamps/data length mismatch: {len(t)} != {len(x)}.")
    if len(t) == 0:
        raise RuntimeError(f"{name} has no samples.")

    order = np.argsort(t)
    t = t[order]
    x = x[order]
    keep = np.ones((len(t),), dtype=bool)
    keep[1:] = np.diff(t) > 1e-9
    t = t[keep]
    x = x[keep]
    if len(t) < 2:
        raise RuntimeError(f"{name} needs at least 2 unique timestamp samples, got {len(t)}.")
    return t, x


def _get_depth_transform(input_res, output_res):
    iw, ih = input_res
    ow, oh = output_res
    input_ratio = iw / ih
    output_ratio = ow / oh

    if input_ratio >= output_ratio:
        rh = oh
        rw = math.ceil(rh / ih * iw)
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)

    w_slice_start = (rw - ow) // 2
    h_slice_start = (rh - oh) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice = slice(h_slice_start, h_slice_start + oh)

    def transform(depth):
        if depth.shape != (ih, iw):
            raise RuntimeError(f"Depth image shape {depth.shape} does not match expected {(ih, iw)}.")
        depth = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_NEAREST)
        depth = depth[h_slice, w_slice]
        return np.ascontiguousarray(depth[..., None].astype(np.float32))

    return transform


class UmiEnv:
    def __init__(self, 
            # required params
            output_dir,
            robot_ip,
            robot_port=8080,
            robot_level=3,
            robot_mode=2,
            robot_joint_dim=7,
            robot_state_read_retries=3,
            robot_state_read_retry_delay=0.01,
            robot_max_consecutive_state_read_failures=10,
            robot_use_udp_state=True,
            robot_udp_port=8888,
            robot_udp_cycle=2,
            robot_udp_target_ip=None,
            robot_udp_state_timeout=0.2,
            robot_launch_timeout=15,
            robot_command_frequency=125,
            robot_command_mode='movep_canfd',
            robot_interpolation_mode='trajectory',
            # env params
            frequency=20,
            # obs
            obs_image_resolution=DEFAULT_RESOLUTION,
            max_obs_buffer_size=60,
            obs_float32=False,
            fisheye_device=DEFAULT_FISHEYE_DEVICE,
            realsense_serial=DEFAULT_REALSENSE_SERIAL,
            gripper_serial_port=DEFAULT_GRIPPER_SERIAL_PORT,
            camera_resolution=DEFAULT_RESOLUTION,
            camera_capture_fps=DEFAULT_FPS,
            fisheye_converter=None,
            # timing
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None,
            debug_get_obs=False
            ):
        if output_dir is None:
            raise ValueError("UmiEnv requires output_dir.")
        if not robot_ip:
            raise ValueError("UmiEnv requires robot_ip for the Realman arm.")
        if robot_port is None:
            raise ValueError("UmiEnv requires robot_port for the Realman arm.")
        if robot_joint_dim is None or robot_joint_dim <= 0:
            raise ValueError(f"UmiEnv requires robot_joint_dim > 0, got {robot_joint_dim}.")
        if robot_state_read_retries <= 0:
            raise ValueError(
                f"UmiEnv requires robot_state_read_retries > 0, got {robot_state_read_retries}.")
        if robot_state_read_retry_delay < 0:
            raise ValueError(
                "UmiEnv requires robot_state_read_retry_delay >= 0, "
                f"got {robot_state_read_retry_delay}.")
        if robot_max_consecutive_state_read_failures <= 0:
            raise ValueError(
                "UmiEnv requires robot_max_consecutive_state_read_failures > 0, "
                f"got {robot_max_consecutive_state_read_failures}.")
        if robot_udp_port <= 0:
            raise ValueError(f"UmiEnv requires robot_udp_port > 0, got {robot_udp_port}.")
        if robot_udp_cycle <= 0:
            raise ValueError(f"UmiEnv requires robot_udp_cycle > 0, got {robot_udp_cycle}.")
        if robot_use_udp_state and robot_udp_target_ip is not None and not robot_udp_target_ip:
            raise ValueError("UmiEnv requires non-empty robot_udp_target_ip when provided.")
        if robot_udp_state_timeout <= 0:
            raise ValueError(
                f"UmiEnv requires robot_udp_state_timeout > 0, got {robot_udp_state_timeout}.")
        if robot_launch_timeout <= 0:
            raise ValueError(f"UmiEnv requires robot_launch_timeout > 0, got {robot_launch_timeout}.")
        if robot_command_frequency <= 0:
            raise ValueError(
                f"UmiEnv requires robot_command_frequency > 0, got {robot_command_frequency}.")
        if robot_command_mode not in ('movep_canfd', 'movep_follow', 'movel', 'movej_p'):
            raise ValueError(
                "UmiEnv requires robot_command_mode to be one of "
                "'movep_canfd', 'movep_follow', 'movel', or 'movej_p', "
                f"got {robot_command_mode!r}.")
        if robot_interpolation_mode not in ('trajectory', 'none'):
            raise ValueError(
                "UmiEnv requires robot_interpolation_mode to be one of "
                f"'trajectory' or 'none', got {robot_interpolation_mode!r}.")
        if not fisheye_device:
            raise ValueError("UmiEnv requires fisheye_device for the Pika fisheye camera.")
        if not realsense_serial:
            raise ValueError("UmiEnv requires realsense_serial for the Pika RealSense camera.")
        if not gripper_serial_port:
            raise ValueError("UmiEnv requires gripper_serial_port for the Pika gripper.")
        if camera_resolution is None or len(camera_resolution) != 2:
            raise ValueError(f"UmiEnv requires camera_resolution=(width,height), got {camera_resolution}.")
        if camera_capture_fps is None or camera_capture_fps <= 0:
            raise ValueError(f"UmiEnv requires camera_capture_fps > 0, got {camera_capture_fps}.")

        output_dir = pathlib.Path(output_dir)
        if not output_dir.parent.is_dir():
            raise ValueError(f"Output directory parent does not exist: {output_dir.parent}")
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        camera_resolution = tuple(camera_resolution)
        obs_image_resolution = tuple(obs_image_resolution)

        rw, rh, col, row = optimal_row_cols(
            n_cameras=2,
            in_wh_ratio=camera_resolution[0] / camera_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )

        rgb_transform = get_image_transform(
            input_res=camera_resolution,
            output_res=obs_image_resolution,
            bgr_to_rgb=False,
        )
        depth_transform = _get_depth_transform(
            input_res=camera_resolution,
            output_res=obs_image_resolution,
        )
        vis_rgb_transform = get_image_transform(
            input_res=camera_resolution,
            output_res=(rw, rh),
            bgr_to_rgb=False,
        )

        def transform(data):
            fisheye = data["fisheye"]
            if fisheye_converter is None:
                fisheye = np.ascontiguousarray(rgb_transform(fisheye))
            else:
                fisheye = np.ascontiguousarray(fisheye_converter.forward(fisheye))

            rgb = np.ascontiguousarray(rgb_transform(data["rgb"]))
            depth = depth_transform(data["depth"])

            if obs_float32:
                fisheye = fisheye.astype(np.float32) / 255.0
                rgb = rgb.astype(np.float32) / 255.0

            data["fisheye"] = fisheye
            data["rgb"] = rgb
            data["depth"] = depth
            return data

        def vis_transform(data):
            fisheye = vis_rgb_transform(data["fisheye"])
            rgb = vis_rgb_transform(data["rgb"])
            return {
                "color": np.stack([fisheye, rgb], axis=0),
                "timestamp": data.get("timestamp", 0.0),
            }

        camera = PikaCamera(
            shm_manager=shm_manager,
            fisheye_device=fisheye_device,
            realsense_serial=realsense_serial,
            resolution=camera_resolution,
            capture_fps=camera_capture_fps,
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=1,
            transform=transform,
            vis_transform=vis_transform if enable_multi_cam_vis else None,
            verbose=False
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=True
            )

        robot = RealmanInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            robot_port=robot_port,
            level=robot_level,
            mode=robot_mode,
            frequency=robot_command_frequency,
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            launch_timeout=robot_launch_timeout,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            receive_latency=robot_obs_latency,
            joint_dim=robot_joint_dim,
            state_read_retries=robot_state_read_retries,
            state_read_retry_delay=robot_state_read_retry_delay,
            max_consecutive_state_read_failures=robot_max_consecutive_state_read_failures,
            use_udp_state=robot_use_udp_state,
            udp_port=robot_udp_port,
            udp_cycle=robot_udp_cycle,
            udp_target_ip=robot_udp_target_ip,
            udp_state_timeout=robot_udp_state_timeout,
            command_mode=robot_command_mode,
            interpolation_mode=robot_interpolation_mode
        )
        gripper = PikaController(
            shm_manager=shm_manager,
            receive_latency=gripper_obs_latency,
            use_meters=True,
            serial_port=gripper_serial_port
        )

        self.camera = camera
        self.robot = robot
        self.gripper = gripper
        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        # timing
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.robot_action_latency = robot_action_latency
        self.gripper_action_latency = gripper_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.debug_get_obs = debug_get_obs
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.camera.is_ready and self.robot.is_ready and self.gripper.is_ready

    def _require_ready(self):
        if not self.is_ready:
            raise RuntimeError(
                "UmiEnv is not ready: "
                f"camera={self.camera.is_ready}, "
                f"realman_arm={self.robot.is_ready}, "
                f"pika_gripper={self.gripper.is_ready}.")
    
    def start(self, wait=True):
        print("[UmiEnv.start] starting camera...", flush=True)
        self.camera.start(wait=False)
        print("[UmiEnv.start] starting gripper...", flush=True)
        self.gripper.start(wait=False)
        print("[UmiEnv.start] starting robot...", flush=True)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            print("[UmiEnv.start] starting multi camera viewer...", flush=True)
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        if self.is_ready:
            self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        print("[UmiEnv.start_wait] waiting for camera...", flush=True)
        self.camera.start_wait()
        print("[UmiEnv.start_wait] camera ready.", flush=True)
        print("[UmiEnv.start_wait] waiting for gripper...", flush=True)
        self.gripper.start_wait()
        print("[UmiEnv.start_wait] gripper ready.", flush=True)
        print("[UmiEnv.start_wait] waiting for robot...", flush=True)
        self.robot.start_wait()
        print("[UmiEnv.start_wait] robot ready.", flush=True)
        if self.multi_cam_vis is not None:
            print("[UmiEnv.start_wait] waiting for multi camera viewer...", flush=True)
            self.multi_cam_vis.start_wait()
            print("[UmiEnv.start_wait] multi camera viewer ready.", flush=True)
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.gripper.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        'current' time is the latest Pika camera timestamp.
        All camera streams use the nearest frame for each requested obs timestamp.
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        self._require_ready()
        if self.debug_get_obs:
            print("[UmiEnv.get_obs] start", flush=True)

        # get data
        # camera_capture_fps Hz, camera_calibrated_timestamp
        k = max(self.camera_obs_horizon, math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (self.camera.capture_fps / self.frequency)))
        if self.debug_get_obs:
            print(f"[UmiEnv.get_obs] reading camera k={k}", flush=True)
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)
        if self.debug_get_obs:
            print("[UmiEnv.get_obs] camera read done", flush=True)

        # 125/500 hz, robot_receive_timestamp
        robot_k = max(2, self.robot_obs_horizon * self.robot_down_sample_steps + 2)
        if self.debug_get_obs:
            print(f"[UmiEnv.get_obs] reading robot k={robot_k}", flush=True)
        if self.robot.ring_buffer.count < robot_k:
            raise RuntimeError(
                f"Not enough Realman state samples: need {robot_k}, "
                f"got {self.robot.ring_buffer.count}.")
        last_robot_data = self.robot.get_state(k=robot_k)
        if self.debug_get_obs:
            print("[UmiEnv.get_obs] robot read done", flush=True)

        # 30 hz, gripper_receive_timestamp
        gripper_k = max(2, self.gripper_obs_horizon * self.gripper_down_sample_steps + 2)
        if self.debug_get_obs:
            print(f"[UmiEnv.get_obs] reading gripper k={gripper_k}", flush=True)
        if self.gripper.ring_buffer.count < gripper_k:
            raise RuntimeError(
                f"Not enough Pika gripper state samples: need {gripper_k}, "
                f"got {self.gripper.ring_buffer.count}.")
        last_gripper_data = self.gripper.get_state(k=gripper_k)
        if self.debug_get_obs:
            print("[UmiEnv.get_obs] gripper read done", flush=True)

        last_timestamp = self.last_camera_data['timestamp'][-1]
        dt = 1 / self.frequency
        if self.debug_get_obs:
            print(f"[UmiEnv.get_obs] aligning obs at timestamp={last_timestamp:.6f}", flush=True)

        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        this_timestamps = self.last_camera_data['timestamp']
        this_idxs = list()
        for t in camera_obs_timestamps:
            nn_idx = np.argmin(np.abs(this_timestamps - t))
            this_idxs.append(nn_idx)
        camera_obs = {
            'fisheye': self.last_camera_data['fisheye'][this_idxs],
            'rgb': self.last_camera_data['rgb'][this_idxs],
            'depth': self.last_camera_data['depth'][this_idxs],
        }

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        robot_t, realman_tcp_poses = _dedupe_time_series(
            last_robot_data['robot_timestamp'],
            last_robot_data['ActualTCPPose'],
            "Realman state")
        pika_gripper_poses = _realman_tcp_pose_to_pika_gripper_pose(realman_tcp_poses)
        if self.debug_get_obs:
            print(
                "[UmiEnv.get_obs] robot timestamp range "
                f"{robot_t[0]:.6f}..{robot_t[-1]:.6f}, "
                f"target {robot_obs_timestamps[0]:.6f}..{robot_obs_timestamps[-1]:.6f}",
                flush=True)
            print("[UmiEnv.get_obs] building robot pose interpolator", flush=True)
        robot_pose_interpolator = PoseInterpolator(
            t=robot_t,
            x=pika_gripper_poses)
        if self.debug_get_obs:
            print("[UmiEnv.get_obs] interpolating robot pose", flush=True)
        pika_gripper_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': pika_gripper_pose[...,:3],
            'robot0_eef_rot_axis_angle': pika_gripper_pose[...,3:]
        }

        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        gripper_t, gripper_pos = _dedupe_time_series(
            last_gripper_data['gripper_timestamp'],
            last_gripper_data['gripper_position'][...,None],
            "Pika gripper state")
        if self.debug_get_obs:
            print(
                "[UmiEnv.get_obs] gripper timestamp range "
                f"{gripper_t[0]:.6f}..{gripper_t[-1]:.6f}, "
                f"target {gripper_obs_timestamps[0]:.6f}..{gripper_obs_timestamps[-1]:.6f}",
                flush=True)
            print("[UmiEnv.get_obs] building gripper interpolator", flush=True)
        gripper_interpolator = get_interp1d(
            t=gripper_t,
            x=gripper_pos
        )
        if self.debug_get_obs:
            print("[UmiEnv.get_obs] interpolating gripper", flush=True)
        gripper_obs = {
            'robot0_gripper_width': gripper_interpolator(gripper_obs_timestamps)
        }
        if self.debug_get_obs:
            print("[UmiEnv.get_obs] alignment done", flush=True)

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': _realman_tcp_pose_to_pika_gripper_pose(
                        last_robot_data['ActualTCPPose']),
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            self.obs_accumulator.put(
                data={
                    'robot0_gripper_width': last_gripper_data['gripper_position'][...,None]
                },
                timestamps=last_gripper_data['gripper_timestamp']
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data['timestamp'] = camera_obs_timestamps

        if self.debug_get_obs:
            print("[UmiEnv.get_obs] done", flush=True)
        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        self._require_ready()
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if actions.ndim != 2 or actions.shape[-1] != 7:
            raise ValueError(f"actions must have shape (N, 7), got {actions.shape}.")
        if timestamps.ndim != 1:
            raise ValueError(f"timestamps must have shape (N,), got {timestamps.shape}.")
        if len(actions) != len(timestamps):
            raise ValueError(
                f"actions and timestamps length mismatch: {len(actions)} != {len(timestamps)}.")

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        r_latency = self.robot_action_latency if compensate_latency else 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0

        # schedule waypoints
        for i in range(len(new_actions)):
            r_actions = _pika_gripper_pose_to_realman_tcp_pose(new_actions[i,:6])
            g_actions = new_actions[i,6:]
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )
            self.gripper.schedule_waypoint(
                pos=g_actions,
                target_time=new_timestamps[i]-g_latency
            )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        self._require_ready()

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        self._require_ready()
        
        # stop video recorder
        self.camera.stop_recording()

        # TODO
        if self.obs_accumulator is not None:
            # recording
            if self.action_accumulator is None:
                raise RuntimeError("obs_accumulator exists but action_accumulator is missing.")

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            if len(action_timestamps) == 0:
                print("Episode discarded: no actions were recorded.")
                self.obs_accumulator = None
                self.action_accumulator = None
                return
            end_time = min(end_time, action_timestamps[-1])

            n_steps = 0
            if np.sum(action_timestamps <= end_time) > 0:
                n_steps = np.nonzero(action_timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)
                episode['robot0_eef_pos'] = robot_pose[:,:3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

                gripper_interpolator = get_interp1d(
                    t=np.array(self.obs_accumulator.timestamps['robot0_gripper_width']),
                    x=np.array(self.obs_accumulator.data['robot0_gripper_width'])
                )
                episode['robot0_gripper_width'] = gripper_interpolator(timestamps)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')


def _parse_six_floats(raw):
    values = raw.replace(",", " ").split()
    if len(values) != 6:
        raise ValueError("Expected 6 numbers: x y z rx ry rz")
    return np.array([float(value) for value in values], dtype=np.float64)


def _parse_one_float(raw):
    values = raw.replace(",", " ").split()
    if len(values) != 1:
        raise ValueError("Expected 1 number.")
    return float(values[0])


def _apply_local_pose_offset(pose, offset):
    return mat_to_pose(pose_to_mat(pose) @ pose_to_mat(offset))


def _latest_robot_pose(env):
    state = env.robot.get_state()
    realman_tcp_pose = np.asarray(state["ActualTCPPose"], dtype=np.float64)
    return _realman_tcp_pose_to_pika_gripper_pose(realman_tcp_pose)


def _latest_joint_pos(env):
    state = env.robot.get_state()
    return np.asarray(state["ActualQ"], dtype=np.float64)


def _latest_gripper_width(env):
    state = env.gripper.get_state()
    return float(state["gripper_position"])


def _submit_action(env, pose, gripper_width, duration):
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}.")
    action = np.zeros((1, 7), dtype=np.float64)
    action[0, :6] = pose
    action[0, 6] = gripper_width
    env.exec_actions(
        actions=action,
        timestamps=np.array([time.time() + duration], dtype=np.float64),
        compensate_latency=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Realman arm + Pika gripper test using UmiEnv.")
    parser.add_argument(
        "-o", "--output",
        default="data_local/pika_env_test",
        help="Directory for replay buffer/video output.")
    parser.add_argument("--robot_ip", default="192.168.1.18")
    parser.add_argument("--robot_port", type=int, default=8080)
    parser.add_argument("--robot_level", type=int, default=3)
    parser.add_argument("--robot_mode", type=int, default=2)
    parser.add_argument("--robot_joint_dim", type=int, default=7)
    parser.add_argument("--robot_state_read_retries", type=int, default=3)
    parser.add_argument("--robot_state_read_retry_delay", type=float, default=0.01)
    parser.add_argument("--robot_max_consecutive_state_read_failures", type=int, default=10)
    parser.add_argument("--no_robot_udp_state", action="store_true", help="Disable Realman UDP state and use TCP polling.")
    parser.add_argument("--robot_udp_port", type=int, default=8888)
    parser.add_argument("--robot_udp_cycle", type=int, default=2, help="Realman UDP push cycle in 5ms units.")
    parser.add_argument("--robot_udp_target_ip", default=None, help="Local PC IP that Realman should push UDP state to.")
    parser.add_argument("--robot_udp_state_timeout", type=float, default=0.2)
    parser.add_argument("--robot_launch_timeout", type=float, default=15.0)
    parser.add_argument("--frequency", type=float, default=20)
    parser.add_argument("--fisheye_device", default=DEFAULT_FISHEYE_DEVICE)
    parser.add_argument("--realsense_serial", default=DEFAULT_REALSENSE_SERIAL)
    parser.add_argument("--gripper_serial_port", default=DEFAULT_GRIPPER_SERIAL_PORT)
    parser.add_argument("--camera_width", type=int, default=DEFAULT_RESOLUTION[0])
    parser.add_argument("--camera_height", type=int, default=DEFAULT_RESOLUTION[1])
    parser.add_argument("--camera_fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--no_vis", action="store_true", help="Disable multi-camera visualizer.")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with SharedMemoryManager() as shm_manager:
        env = UmiEnv(
            output_dir=output_dir,
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            robot_level=args.robot_level,
            robot_mode=args.robot_mode,
            robot_joint_dim=args.robot_joint_dim,
            robot_state_read_retries=args.robot_state_read_retries,
            robot_state_read_retry_delay=args.robot_state_read_retry_delay,
            robot_max_consecutive_state_read_failures=args.robot_max_consecutive_state_read_failures,
            robot_use_udp_state=not args.no_robot_udp_state,
            robot_udp_port=args.robot_udp_port,
            robot_udp_cycle=args.robot_udp_cycle,
            robot_udp_target_ip=args.robot_udp_target_ip,
            robot_udp_state_timeout=args.robot_udp_state_timeout,
            robot_launch_timeout=args.robot_launch_timeout,
            frequency=args.frequency,
            fisheye_device=args.fisheye_device,
            realsense_serial=args.realsense_serial,
            gripper_serial_port=args.gripper_serial_port,
            camera_resolution=(args.camera_width, args.camera_height),
            camera_capture_fps=args.camera_fps,
            enable_multi_cam_vis=not args.no_vis,
            shm_manager=shm_manager,
        )

        init_pose = None
        try:
            env.start(wait=True)
            init_pose = _latest_robot_pose(env)
            print("Ready.")
            print("Initial Pika gripper pose:", init_pose.tolist())

            while True:
                print("\nSelect next action:")
                print("1. get state")
                print("2. move absolute pose")
                print("3. move offset from recorded init pose")
                print("4. record init pose")
                print("5. get gripper")
                print("6. set gripper width")
                print("7. exit")
                command = input("Enter command: ").strip().lower()

                if command in ("1", "get state", "get_state"):
                    pose = _latest_robot_pose(env)
                    joint = _latest_joint_pos(env)
                    width_m = _latest_gripper_width(env)
                    print("pika_gripper_pose:", pose.tolist())
                    print("joint:", joint.tolist())
                    print(f"gripper_width: {width_m:.6f} m ({width_m * 1000:.2f} mm)")

                elif command in ("2", "move absolute pose", "move_absolute", "absolute"):
                    raw_pose = input("Enter absolute Pika gripper pose [x y z rx ry rz]: ").strip()
                    pose = _parse_six_floats(raw_pose)
                    width_m = _latest_gripper_width(env)
                    _submit_action(env, pose, width_m, args.duration)
                    print(f"Submitted absolute Pika gripper pose for t+{args.duration:.2f}s:", pose.tolist())

                elif command in ("3", "move offset from recorded init pose", "move offset", "offset"):
                    if init_pose is None:
                        raise RuntimeError("No init pose recorded.")
                    raw_delta = input("Enter local Pika gripper offset [dx dy dz drx dry drz]: ").strip()
                    delta = _parse_six_floats(raw_delta)
                    pose = _apply_local_pose_offset(init_pose, delta)
                    width_m = _latest_gripper_width(env)
                    _submit_action(env, pose, width_m, args.duration)
                    print(f"Submitted init-pose local Pika gripper offset for t+{args.duration:.2f}s:", pose.tolist())

                elif command in ("4", "record init pose", "record_init_pose", "record init"):
                    init_pose = _latest_robot_pose(env)
                    print("Recorded init Pika gripper pose:", init_pose.tolist())

                elif command in ("5", "get gripper", "get_gripper"):
                    width_m = _latest_gripper_width(env)
                    print(f"gripper_width: {width_m:.6f} m ({width_m * 1000:.2f} mm)")

                elif command in ("6", "set gripper width", "set_gripper_width", "set gripper"):
                    raw_width = input("Enter gripper width in mm: ").strip()
                    width_mm = _parse_one_float(raw_width)
                    width_m = width_mm / 1000.0
                    pose = _latest_robot_pose(env)
                    _submit_action(env, pose, width_m, args.duration)
                    print(f"Submitted gripper width for t+{args.duration:.2f}s: {width_mm:.2f} mm")

                elif command in ("7", "exit", "quit", "q"):
                    break

                else:
                    print("Unknown command. Use 1, 2, 3, 4, 5, 6, or 7.")

        finally:
            env.stop(wait=True)


if __name__ == "__main__":
    main()
