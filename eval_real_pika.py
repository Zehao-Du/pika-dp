"""
Usage:
(zehao): python eval_real_pika.py -i outputs/checkpoints/collect_blocks/0703/epoch=0100-train_loss=0.011.ckpt -o data_local/pika_eval

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 
"""
CLOSE_THRESHOOD = 0.065
# %%
import sys
import os
import functools
import http.server

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import pathlib
import threading
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as st
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.pika_env import (
    DEFAULT_FISHEYE_DEVICE,
    DEFAULT_FPS,
    DEFAULT_GRIPPER_SERIAL_PORT,
    DEFAULT_REALSENSE_SERIAL,
    DEFAULT_RESOLUTION,
    UmiEnv,
    _pika_gripper_pose_to_realman_tcp_pose,
)
from umi.real_world.real_inference_util import (get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.common.pose_util import pose10d_to_mat, mat_to_pose

OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_image_obs_horizon(shape_meta):
    horizons = [
        attr["horizon"]
        for attr in shape_meta["obs"].values()
        if attr.get("type") in ("rgb", "depth")
    ]
    if not horizons:
        raise RuntimeError("shape_meta.obs must contain at least one rgb/depth observation.")
    return max(horizons)


def get_obs_horizon(shape_meta, key):
    try:
        return shape_meta["obs"][key]["horizon"]
    except KeyError as e:
        raise RuntimeError(f"shape_meta.obs is missing required key: {key}") from e


def get_episode_start_pose_from_obs(obs):
    return [
        np.concatenate([
            obs['robot0_eef_pos'][-1],
            obs['robot0_eef_rot_axis_angle'][-1]
        ], axis=-1)
    ]


def print_action_debug(
        raw_action,
        action,
        timestamps,
        obs_timestamps,
        label,
        scheduled_mask=None,
        obs=None,
        true_label="exec",
        false_label="skip"):
    raw_action = np.asarray(raw_action)
    if raw_action.ndim != 2 or raw_action.shape[-1] != 10:
        print(f"[ActionDebug:{label}] unexpected raw action shape: {raw_action.shape}")
        return
    action = np.asarray(action)
    if action.ndim != 2 or action.shape[-1] != 7:
        print(f"[ActionDebug:{label}] unexpected absolute action shape: {action.shape}")
        return
    if len(action) != len(raw_action):
        print(f"[ActionDebug:{label}] raw/action length mismatch: {len(raw_action)} != {len(action)}")
        return

    rel_pose = mat_to_pose(pose10d_to_mat(raw_action[:, :9]))
    rel_gripper = raw_action[:, 9]
    robot_abs_pose = np.stack([
        _pika_gripper_pose_to_realman_tcp_pose(x[:6])
        for x in action
    ], axis=0)
    robot_abs_gripper = action[:, 6]
    if scheduled_mask is None:
        scheduled_mask = np.ones((len(action),), dtype=bool)
    else:
        scheduled_mask = np.asarray(scheduled_mask, dtype=bool)
        if scheduled_mask.shape != (len(action),):
            print(f"[ActionDebug:{label}] unexpected scheduled_mask shape: {scheduled_mask.shape}")
            return

    colors = {
        "header": "\033[95m",
        "step": "\033[90m",
        "pos": "\033[92m",
        "rot": "\033[96m",
        "gripper": "\033[93m",
        "robot": "\033[94m",
        "skip": "\033[91m",
        "reset": "\033[0m",
    }
    horizon = len(raw_action)
    print(
        f"{colors['header']}[ActionDebug:{label}] model relative + robot absolute steps 1-{horizon - 1} "
        f"(h={horizon}){colors['reset']}")
    for step_idx in range(1, horizon):
        status = true_label if scheduled_mask[step_idx] else false_label
        status_color = colors["step"] if scheduled_mask[step_idx] else colors["skip"]
        rel_pos = np.array2string(rel_pose[step_idx, :3], precision=4, suppress_small=True)
        rel_rot = np.array2string(rel_pose[step_idx, 3:6], precision=4, suppress_small=True)
        rel_g = f"{rel_gripper[step_idx]:.4f}"
        abs_pos = np.array2string(robot_abs_pose[step_idx, :3], precision=4, suppress_small=True)
        abs_rot = np.array2string(robot_abs_pose[step_idx, 3:6], precision=4, suppress_small=True)
        abs_g = f"{robot_abs_gripper[step_idx]:.4f}"
        print(
            f"{colors['step']}step {step_idx:02d}{colors['reset']} "
            f"{status_color}[{status}]{colors['reset']} | "
            f"model_relative "
            f"{colors['pos']}pos: {rel_pos}{colors['reset']} "
            f"{colors['rot']}rot: {rel_rot}{colors['reset']} "
            f"{colors['gripper']}gripper: {rel_g}{colors['reset']} | "
            f"{colors['robot']}robot_absolute_realman_tcp "
            f"pos: {abs_pos} rot: {abs_rot} gripper: {abs_g}{colors['reset']}")


def prompt_human_action_step(horizon):
    while True:
        value = input(f"Execute action step [1-{horizon - 1}], or 0 to skip: ").strip()
        try:
            step_idx = int(value)
        except ValueError:
            print(f"Invalid input {value!r}; enter an integer from 0 to {horizon - 1}.")
            continue
        if 0 <= step_idx < horizon:
            return step_idx
        print(f"Step {step_idx} is out of range; enter 0 to {horizon - 1}.")


def prompt_human_inference():
    value = input("Press Enter to run inference at current pose, or q to end episode: ").strip().lower()
    return value not in ("q", "quit", "exit")


def parse_action_execution_mode(value):
    value = str(value).strip().lower()
    if value in ('all', 'last', 'first', 'human'):
        return value, None
    try:
        fixed_step = int(value)
    except ValueError as e:
        raise click.BadParameter(
            "must be one of all/last/first/human, or a positive integer step index."
        ) from e
    if fixed_step <= 0:
        raise click.BadParameter(
            "fixed action step must be >= 1; use human mode and input 0 to skip.")
    return 'step', fixed_step


def apply_gripper_width_rule(action, close_threshold=CLOSE_THRESHOOD, close_delta=0.01):
    action = np.asarray(action).copy()
    if action.ndim != 2 or action.shape[-1] != 7:
        raise RuntimeError(f"Expected env action shape (N, 7), got {action.shape}.")
    action[action[:, 6] < close_threshold, 6] -= close_delta
    return action


def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta
    return delta


def apply_table_collision_rule(action, table_height, enabled=True):
    action = np.asarray(action).copy()
    if not enabled:
        return action
    if action.ndim != 2 or action.shape[-1] != 7:
        raise RuntimeError(f"Expected env action shape (N, 7), got {action.shape}.")
    max_delta = 0.0
    for target in action:
        max_delta = max(
            max_delta,
            solve_table_collision(
                ee_pose=target[:6],
                gripper_width=target[6],
                height_threshold=table_height))
    if max_delta > 0:
        print(f"[TableCollision] lifted action z by up to {max_delta:.4f} m; table_height={table_height:.4f}", flush=True)
    return action


def _to_uint8_image(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def dump_obs_debug(debug_obs_dir, obs, obs_dict_np, step_idx):
    if not debug_obs_dir:
        return
    out_dir = pathlib.Path(debug_obs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"obs_{step_idx:06d}"

    fisheye = _to_uint8_image(obs["fisheye"][-1])
    rgb = _to_uint8_image(obs["rgb"][-1])
    cv2.imwrite(str(prefix) + "_fisheye.png", cv2.cvtColor(fisheye, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(prefix) + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    depth = np.asarray(obs_dict_np["depth"][-1, 0], dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    depth_vis = np.clip(depth, 0.0, 0.5) / 0.5
    depth_vis = (depth_vis * 255.0).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_vis[~valid] = (0, 0, 0)
    cv2.imwrite(str(prefix) + "_depth.png", depth_vis)

    stats = {
        "timestamp": np.asarray(obs["timestamp"]),
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"]),
        "robot0_eef_rot_axis_angle": np.asarray(obs["robot0_eef_rot_axis_angle"]),
        "robot0_gripper_width": np.asarray(obs["robot0_gripper_width"]),
        "depth_min": np.asarray(float(np.min(depth[valid])) if np.any(valid) else np.nan),
        "depth_median": np.asarray(float(np.median(depth[valid])) if np.any(valid) else np.nan),
        "depth_max": np.asarray(float(np.max(depth[valid])) if np.any(valid) else np.nan),
        "depth_valid_ratio": np.asarray(float(np.mean(valid))),
    }
    np.savez(str(prefix) + "_stats.npz", **stats)
    print(
        "[ObsDebug] "
        f"saved {prefix.name}; depth valid={float(np.mean(valid)):.3f}, "
        f"median={float(np.median(depth[valid])) if np.any(valid) else np.nan:.4f}m, "
        f"eef_pos={np.array2string(obs['robot0_eef_pos'][-1], precision=4, suppress_small=True)}, "
        f"gripper={float(obs['robot0_gripper_width'][-1, 0]):.4f}",
        flush=True)


def make_policy_obs_canvas(obs, obs_dict_np):
    tile_w, tile_h = 426, 320
    fisheye = cv2.resize(
        cv2.cvtColor(_to_uint8_image(obs["fisheye"][-1]), cv2.COLOR_RGB2BGR),
        (tile_w, tile_h),
        interpolation=cv2.INTER_AREA)
    rgb = cv2.resize(
        cv2.cvtColor(_to_uint8_image(obs["rgb"][-1]), cv2.COLOR_RGB2BGR),
        (tile_w, tile_h),
        interpolation=cv2.INTER_AREA)

    depth = np.asarray(obs_dict_np["depth"][-1, 0], dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    depth_vis = np.clip(depth, 0.0, 0.5) / 0.5
    depth_vis = (depth_vis * 255.0).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_vis[~valid] = (0, 0, 0)
    depth_vis = cv2.resize(depth_vis, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

    for name, img in (("fisheye", fisheye), ("rgb", rgb), ("depth", depth_vis)):
        cv2.putText(img, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

    pika_pose = np.concatenate([
        obs["robot0_eef_pos"][-1],
        obs["robot0_eef_rot_axis_angle"][-1],
    ], axis=-1)
    realman_tcp_pose = _pika_gripper_pose_to_realman_tcp_pose(pika_pose)
    gripper_width = float(obs["robot0_gripper_width"][-1, 0])

    info_h = 150
    info = np.zeros((info_h, tile_w * 3, 3), dtype=np.uint8)
    lines = [
        "absolute Realman TCP pose",
        "pos: " + np.array2string(realman_tcp_pose[:3], precision=4, suppress_small=True),
        "rot: " + np.array2string(realman_tcp_pose[3:6], precision=4, suppress_small=True),
        "absolute Pika gripper pose",
        "pos: " + np.array2string(pika_pose[:3], precision=4, suppress_small=True),
        "rot: " + np.array2string(pika_pose[3:6], precision=4, suppress_small=True),
        f"gripper_width: {gripper_width:.4f} m",
        f"depth_valid: {float(np.mean(valid)):.3f}, depth_median: "
        f"{float(np.median(depth[valid])) if np.any(valid) else np.nan:.4f} m",
    ]
    x_positions = [12, tile_w + 12, tile_w * 2 + 12]
    text_groups = [lines[:3], lines[3:6], lines[6:]]
    for x, group in zip(x_positions, text_groups):
        y = 28
        for line in group:
            cv2.putText(info, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1, cv2.LINE_AA)
            y += 30

    return np.vstack([np.hstack([fisheye, rgb, depth_vis]), info])


class _PolicyObsRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def _write_policy_obs_index(out_dir):
    index_html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pika Policy Obs</title>
  <style>
    html, body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    header { padding: 10px 14px; font-size: 14px; background: #222; }
    img { display: block; width: 100vw; height: auto; image-rendering: auto; }
  </style>
</head>
<body>
  <header>Live policy observation. Refreshes from latest.jpg.</header>
  <img id="latest" src="latest.jpg" alt="waiting for first policy observation">
  <script>
    const img = document.getElementById("latest");
    setInterval(() => { img.src = "latest.jpg?t=" + Date.now(); }, 100);
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")


def start_policy_obs_window(enabled, window_name="Policy obs"):
    if not enabled:
        return None
    out_dir = pathlib.Path("/tmp/pika_policy_obs_window")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_policy_obs_index(out_dir)
    handler = functools.partial(_PolicyObsRequestHandler, directory=str(out_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[PolicyObsWindow] enabled: open {url}", flush=True)
    return {
        "server": server,
        "thread": thread,
        "latest_path": out_dir / "latest.jpg",
        "tmp_path": out_dir / "latest.tmp.jpg",
        "url": url,
    }


def stop_policy_obs_window(viewer):
    if viewer is None:
        return
    try:
        viewer["server"].shutdown()
        viewer["server"].server_close()
    except BaseException:
        pass


def show_policy_obs_window(obs, obs_dict_np, viewer=None):
    if viewer is None:
        return
    canvas = make_policy_obs_canvas(obs, obs_dict_np)
    ok = cv2.imwrite(str(viewer["tmp_path"]), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write policy obs image to {viewer['tmp_path']}.")
    os.replace(viewer["tmp_path"], viewer["latest_path"])


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', default='data_local/pika_eval', help='Directory to save recording')
@click.option('--robot_ip', default='192.168.1.18')
@click.option('--robot_port', default=8080, type=int)
@click.option('--robot_level', default=3, type=int)
@click.option('--robot_mode', default=2, type=int)
@click.option('--robot_joint_dim', default=7, type=int)
@click.option('--robot_state_read_retries', default=3, type=int)
@click.option('--robot_state_read_retry_delay', default=0.01, type=float)
@click.option('--robot_max_consecutive_state_read_failures', default=10, type=int)
@click.option('--no_robot_udp_state', is_flag=True, default=False)
@click.option('--robot_udp_port', default=8888, type=int)
@click.option('--robot_udp_cycle', default=2, type=int)
@click.option('--robot_udp_target_ip', default=None, help='Local PC IP that Realman should push UDP state to.')
@click.option('--robot_udp_state_timeout', default=0.2, type=float)
@click.option('--robot_launch_timeout', default=15.0, type=float)
@click.option('--robot_command_frequency', default=125, type=float, help='Realman low-level command frequency in Hz.')
@click.option('--robot_command_mode', default='movep_canfd',
              type=click.Choice(['movep_canfd', 'movep_follow', 'movel', 'movej_p']),
              help='Realman low-level pose command API.')
@click.option('--robot_interpolation_mode', default='trajectory',
              type=click.Choice(['trajectory', 'none']),
              help='Realman controller interpolation mode.')
@click.option('--fisheye_device', default=DEFAULT_FISHEYE_DEVICE)
@click.option('--realsense_serial', default=DEFAULT_REALSENSE_SERIAL)
@click.option('--gripper_serial_port', default=DEFAULT_GRIPPER_SERIAL_PORT)
@click.option('--camera_width', default=DEFAULT_RESOLUTION[0], type=int)
@click.option('--camera_height', default=DEFAULT_RESOLUTION[1], type=int)
@click.option('--camera_fps', default=DEFAULT_FPS, type=int)
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--action_execution_mode', default='all',
              help="Which policy action steps to submit: all, last, first, human, or a fixed step index like 8.")
@click.option('--num_inference_steps', default=16, type=int, help="DDIM/DDPM denoising iterations per policy call.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--debug_obs_dir', default=None, help='Optional directory to save policy input obs debug frames.')
@click.option('--policy_obs_window/--no_policy_obs_window', default=True,
              help='Show a browser page with policy RGB/depth obs and absolute TCP pose.')
@click.option('--table_height', default=0.23, type=float,
              help='Absolute table height in the Pika gripper/world z axis, in meters.')
@click.option('--disable_table_collision', is_flag=True, default=False,
              help='Disable UMI-style table collision lifting before executing actions.')
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)

def main(input, output, robot_ip, robot_port, robot_level, robot_mode, robot_joint_dim,
    robot_state_read_retries, robot_state_read_retry_delay,
    robot_max_consecutive_state_read_failures, no_robot_udp_state,
    robot_udp_port, robot_udp_cycle, robot_udp_target_ip, robot_udp_state_timeout,
    robot_launch_timeout, robot_command_frequency, robot_command_mode, robot_interpolation_mode,
    fisheye_device, realsense_serial, gripper_serial_port,
    camera_width, camera_height, camera_fps,
    match_dataset, match_episode, match_camera,
    steps_per_inference, action_execution_mode, num_inference_steps, max_duration,
    frequency, debug_obs_dir, policy_obs_window, table_height, disable_table_collision,
    sim_fov, camera_intrinsics):
    action_execution_mode, fixed_action_step = parse_action_execution_mode(action_execution_mode)
    
    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    cfg.policy.obs_encoder.pretrained = False
    if "frozen" in cfg.policy.obs_encoder:
        cfg.policy.obs_encoder.frozen = False
    print("Using local checkpoint weights only; skipping cloud timm pretrained lookup.", flush=True)
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    with SharedMemoryManager() as shm_manager:
        with UmiEnv(
                output_dir=output, 
                robot_ip=robot_ip,
                robot_port=robot_port,
                robot_level=robot_level,
                robot_mode=robot_mode,
                robot_joint_dim=robot_joint_dim,
                robot_state_read_retries=robot_state_read_retries,
                robot_state_read_retry_delay=robot_state_read_retry_delay,
                robot_max_consecutive_state_read_failures=robot_max_consecutive_state_read_failures,
                robot_use_udp_state=not no_robot_udp_state,
                robot_udp_port=robot_udp_port,
                robot_udp_cycle=robot_udp_cycle,
                robot_udp_target_ip=robot_udp_target_ip,
                robot_udp_state_timeout=robot_udp_state_timeout,
                robot_launch_timeout=robot_launch_timeout,
                robot_command_frequency=robot_command_frequency,
                robot_command_mode=robot_command_mode,
                robot_interpolation_mode=robot_interpolation_mode,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                fisheye_device=fisheye_device,
                realsense_serial=realsense_serial,
                gripper_serial_port=gripper_serial_port,
                camera_resolution=(camera_width, camera_height),
                camera_capture_fps=camera_fps,
                enable_multi_cam_vis=False,
                
                # latency
                # camera_obs_latency=0.17,
                # robot_obs_latency=0.0001,
                # gripper_obs_latency=0.01,
                # robot_action_latency=0.18,
                # gripper_action_latency=0.1,
                camera_obs_latency=0.0,
                robot_obs_latency=0.0,
                gripper_obs_latency=0.0,
                robot_action_latency=0.0,
                gripper_action_latency=0.0,

                # obs
                camera_obs_horizon=get_image_obs_horizon(cfg.task.shape_meta),
                robot_obs_horizon=get_obs_horizon(cfg.task.shape_meta, "robot0_eef_pos"),
                gripper_obs_horizon=get_obs_horizon(cfg.task.shape_meta, "robot0_gripper_width"),
                fisheye_converter=fisheye_converter,
                # action
                max_pos_speed=0.05,
                max_rot_speed=0.15,
                debug_get_obs=False,
                shm_manager=shm_manager) as env:
            policy_obs_viewer = start_policy_obs_window(policy_obs_window)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break
                        # img = VideoFileClip(str(match_video_path)).get_frame(0)

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = num_inference_steps
            policy.debug_predict_action = False
            print("num_inference_steps:", policy.num_inference_steps)
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference", flush=True)
            s_warmup = time.time()
            print("Warmup: getting env obs...", flush=True)
            obs = env.get_obs()
            print(f"Warmup: got env obs in {time.time() - s_warmup:.3f}s", flush=True)
            with torch.no_grad():
                s = time.time()
                print("Warmup: resetting policy...", flush=True)
                policy.reset()
                print(f"Warmup: policy reset in {time.time() - s:.3f}s", flush=True)
                s = time.time()
                print("Warmup: building obs dict...", flush=True)
                warmup_episode_start_pose = get_episode_start_pose_from_obs(obs)
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=warmup_episode_start_pose)
                show_policy_obs_window(obs, obs_dict_np, viewer=policy_obs_viewer)
                print(f"Warmup: built obs dict in {time.time() - s:.3f}s", flush=True)
                s = time.time()
                print("Warmup: moving obs tensors to CUDA...", flush=True)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                print(f"Warmup: moved obs tensors to CUDA in {time.time() - s:.3f}s", flush=True)
                s = time.time()
                print("Warmup: running policy.predict_action...", flush=True)
                result = policy.predict_action(obs_dict)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"Warmup: policy.predict_action finished in {time.time() - s:.3f}s", flush=True)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7
                del result

            print('Ready!')
            while True:
                print("Policy in control!")
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    episode_start_pose = None
                    while True:
                        if action_execution_mode == 'human':
                            if not prompt_human_inference():
                                env.end_episode()
                                break

                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        if episode_start_pose is None:
                            episode_start_pose = get_episode_start_pose_from_obs(obs)
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            print("Inference: building obs dict...", flush=True)
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                            obs_pose_repr=obs_pose_rep,
                            episode_start_pose=episode_start_pose)
                            dump_obs_debug(debug_obs_dir, obs, obs_dict_np, iter_idx)
                            show_policy_obs_window(obs, obs_dict_np, viewer=policy_obs_viewer)
                            print("Inference: moving obs tensors to CUDA...", flush=True)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            print("Inference: running policy.predict_action...", flush=True)
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            if raw_action.shape[-1] != 10:
                                raise RuntimeError(f"Policy action must have 10 dims, got {raw_action.shape}.")
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            if action.shape[-1] != 7:
                                raise RuntimeError(f"Converted env action must have 7 dims, got {action.shape}.")
                            action = apply_gripper_width_rule(action)
                            action = apply_table_collision_rule(
                                action,
                                table_height=table_height,
                                enabled=not disable_table_collision)
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        this_target_poses = action

                        # deal with timing
                        # the same step actions are always the target for
                        horizon = len(raw_action)
                        action_timestamps = (np.arange(horizon, dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if action_execution_mode == 'human':
                            valid_mask = np.ones_like(is_new, dtype=bool)
                            valid_mask[0] = False
                            print_action_debug(
                                raw_action,
                                action,
                                action_timestamps,
                                obs_timestamps,
                                "predicted",
                                scheduled_mask=valid_mask,
                                obs=obs,
                                true_label="valid",
                                false_label="skip")
                            selected_step = prompt_human_action_step(horizon)
                            if selected_step == 0:
                                this_target_poses = action[:0]
                                action_timestamps = np.array([], dtype=np.float64)
                                scheduled_mask = np.zeros_like(is_new, dtype=bool)
                                print("Human selected 0; skipping this action cycle.", flush=True)
                            else:
                                curr_time = time.time()
                                next_step_idx = int(np.ceil(
                                    (curr_time + action_exec_latency - eval_t_start) / dt))
                                action_timestamp = eval_t_start + next_step_idx * dt
                                this_target_poses = action[[selected_step]]
                                action_timestamps = np.array([action_timestamp], dtype=np.float64)
                                scheduled_mask = np.zeros_like(is_new, dtype=bool)
                                scheduled_mask[selected_step] = True
                                print_action_debug(
                                    raw_action,
                                    action,
                                    action_timestamps,
                                    obs_timestamps,
                                    "human_selected",
                                    scheduled_mask=scheduled_mask,
                                    obs=obs)
                        elif action_execution_mode == 'step':
                            if fixed_action_step >= horizon:
                                raise RuntimeError(
                                    f"--action_execution_mode {fixed_action_step} is out of range "
                                    f"for policy horizon {horizon}; valid fixed steps are 1..{horizon - 1}.")
                            scheduled_indices = np.array([fixed_action_step], dtype=np.int64)
                            action_timestamp = action_timestamps[fixed_action_step]
                            if action_timestamp <= (curr_time + action_exec_latency):
                                next_step_idx = int(np.ceil(
                                    (curr_time + action_exec_latency - eval_t_start) / dt))
                                action_timestamp = eval_t_start + next_step_idx * dt
                                print(
                                    f"Fixed step {fixed_action_step} over budget; "
                                    f"rescheduled in {action_timestamp - curr_time:.3f}s")
                            action_timestamps = np.array([action_timestamp], dtype=np.float64)
                            scheduled_mask = np.zeros_like(is_new, dtype=bool)
                            scheduled_mask[fixed_action_step] = True
                        elif np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            scheduled_indices = np.array([horizon - 1], dtype=np.int64)
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                            scheduled_mask = np.zeros_like(is_new, dtype=bool)
                            scheduled_mask[-1] = True
                        else:
                            scheduled_indices = np.flatnonzero(is_new)
                            if action_execution_mode == 'last':
                                scheduled_indices = scheduled_indices[[-1]]
                            elif action_execution_mode == 'first':
                                scheduled_indices = scheduled_indices[[0]]
                            action_timestamps = action_timestamps[scheduled_indices]
                            scheduled_mask = np.zeros_like(is_new, dtype=bool)
                            scheduled_mask[scheduled_indices] = True
                        if action_execution_mode != 'human':
                            this_target_poses = action[scheduled_indices]
                            print_action_debug(
                                raw_action,
                                action,
                                action_timestamps,
                                obs_timestamps,
                                "predicted",
                                scheduled_mask=scheduled_mask,
                                obs=obs)
                        # execute actions
                        if len(this_target_poses) > 0:
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True
                            )
                        print(f"Submitted {len(this_target_poses)} steps of actions.", flush=True)

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            env.end_episode()
                            break

                        if action_execution_mode == 'human':
                            iter_idx += steps_per_inference
                            continue

                        # wait for execution
                        wait_until = t_cycle_end - frame_latency
                        wait_sec = wait_until - time.monotonic()
                        print(f"Post-action: waiting {max(wait_sec, 0.0):.3f}s until next inference cycle.", flush=True)
                        precise_wait(wait_until)
                        print("Post-action: cycle done.", flush=True)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
