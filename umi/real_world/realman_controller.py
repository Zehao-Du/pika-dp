import os
import sys
import time
import enum
import traceback
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from ctypes import CFUNCTYPE
import threading
import socket
import numpy as np
import scipy.spatial.transform as st

try:
    from umi.shared_memory.shared_memory_queue import (
        SharedMemoryQueue, Empty)
    from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
except ImportError as e:
    raise ImportError(
        "Missing dependency for UMI shared memory. Install required packages "
        "in the active Python environment, for example: uv pip install atomics"
    ) from e
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait

DEFAULT_IP = "192.168.1.18"
DEFAULT_PORT = 8080
DEFAULT_LEVEL = 3
DEFAULT_MODE = 2
CONTROL_TOOLS_DIR = "/home/ubuntu/Documents/CodeField/zehao/Control_Tools"
CONTROL_TOOLS_RM_API_DIR = os.path.join(CONTROL_TOOLS_DIR, "RM_API2", "Python")

for path in (CONTROL_TOOLS_DIR, CONTROL_TOOLS_RM_API_DIR):
    if path not in sys.path:
        sys.path.append(path)

try:
    from Robotic_Arm.rm_robot_interface import (
        RoboticArm,
        rm_realtime_arm_joint_state_t,
        rm_realtime_push_config_t,
        rm_thread_mode_e,
    )
except ImportError as e:
    raise ImportError(
        "Missing dependency: Realman SDK Robotic_Arm is required for "
        "RealmanInterpolationController. Expected it under "
        f"{CONTROL_TOOLS_RM_API_DIR}, or installed in the active Python environment."
    ) from e


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


def _make_receive_example(receive_keys, joint_dim):
    example = dict()
    for key in receive_keys:
        if key in ('ActualTCPPose', 'ActualTCPSpeed', 'TargetTCPPose', 'TargetTCPSpeed'):
            example[key] = np.zeros((6,), dtype=np.float64)
        elif key in ('ActualQ', 'ActualQd', 'TargetQ', 'TargetQd'):
            example[key] = np.zeros((joint_dim,), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported receive key for Realman controller: {key}")
    example['robot_receive_timestamp'] = time.time()
    example['robot_timestamp'] = time.time()
    return example


def _check_vector(name, value, shape):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape:
        raise RuntimeError(f"Expected {name} shape {shape}, got {value.shape}")
    return value


def _infer_local_ip_for_target(target_ip):
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((target_ip, 9))
        return sock.getsockname()[0]
    except OSError as e:
        raise RuntimeError(
            f"Failed to infer local UDP target IP for Realman robot {target_ip}. "
            "Pass robot_udp_target_ip explicitly."
        ) from e
    finally:
        if sock is not None:
            sock.close()


def _realman_euler_pose_to_rotvec_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    out = np.array(pose, dtype=np.float64, copy=True)
    out[3:] = st.Rotation.from_euler('xyz', pose[3:]).as_rotvec()
    return out


def _rotvec_pose_to_realman_euler_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    out = np.array(pose, dtype=np.float64, copy=True)
    out[3:] = st.Rotation.from_rotvec(pose[3:]).as_euler('xyz')
    return out


class RealmanInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip=DEFAULT_IP, robot_port=DEFAULT_PORT, level=DEFAULT_LEVEL, mode=DEFAULT_MODE,
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=15,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=None,
            receive_latency=0.0,
            joint_dim=7,
            connect_retry_count=30,
            connect_retry_interval=0.5,
            command_mode='movep_canfd',
            interpolation_mode='trajectory',
            follow=True,
            trajectory_mode=0,
            radio=0,
            move_speed=5,
            move_radius=0,
            move_connect=1,
            move_block=0,
            state_read_retries=3,
            state_read_retry_delay=0.01,
            max_consecutive_state_read_failures=10,
            use_udp_state=True,
            udp_port=8888,
            udp_cycle=2,
            udp_target_ip=None,
            udp_state_timeout=0.2
            ):
        """
        frequency: control command frequency.
        command_mode: one of 'movep_canfd', 'movep_follow', 'movel', or 'movej_p'.
        max_pos_speed: m/s
        max_rot_speed: rad/s
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """
        # verify
        if shm_manager is None:
            raise ValueError("RealmanInterpolationController requires shm_manager.")
        if not robot_ip:
            raise ValueError("RealmanInterpolationController requires robot_ip.")
        if robot_port is None:
            raise ValueError("RealmanInterpolationController requires robot_port.")
        if level is None:
            raise ValueError("RealmanInterpolationController requires level.")
        if mode is None:
            raise ValueError("RealmanInterpolationController requires mode.")
        if not 0 < frequency <= 500:
            raise ValueError(f"frequency must be in (0, 500], got {frequency}.")
        if not 0.03 <= lookahead_time <= 0.2:
            raise ValueError(f"lookahead_time must be in [0.03, 0.2], got {lookahead_time}.")
        if not 100 <= gain <= 2000:
            raise ValueError(f"gain must be in [100, 2000], got {gain}.")
        if max_pos_speed <= 0:
            raise ValueError(f"max_pos_speed must be > 0, got {max_pos_speed}.")
        if max_rot_speed <= 0:
            raise ValueError(f"max_rot_speed must be > 0, got {max_rot_speed}.")
        if joint_dim <= 0:
            raise ValueError(f"joint_dim must be > 0, got {joint_dim}.")
        if command_mode not in ('movep_canfd', 'movep_follow', 'movel', 'movej_p'):
            raise ValueError(
                "command_mode must be one of 'movep_canfd', 'movep_follow', "
                "'movel', or 'movej_p', "
                f"got {command_mode!r}.")
        if interpolation_mode not in ('trajectory', 'none'):
            raise ValueError(
                "interpolation_mode must be one of 'trajectory' or 'none', "
                f"got {interpolation_mode!r}.")
        if state_read_retries <= 0:
            raise ValueError(f"state_read_retries must be > 0, got {state_read_retries}.")
        if state_read_retry_delay < 0:
            raise ValueError(f"state_read_retry_delay must be >= 0, got {state_read_retry_delay}.")
        if max_consecutive_state_read_failures <= 0:
            raise ValueError(
                "max_consecutive_state_read_failures must be > 0, "
                f"got {max_consecutive_state_read_failures}.")
        if udp_port <= 0:
            raise ValueError(f"udp_port must be > 0, got {udp_port}.")
        if udp_cycle <= 0:
            raise ValueError(f"udp_cycle must be > 0, got {udp_cycle}.")
        if use_udp_state and udp_target_ip is None:
            udp_target_ip = _infer_local_ip_for_target(robot_ip)
        if use_udp_state and not udp_target_ip:
            raise ValueError("RealmanInterpolationController requires udp_target_ip when UDP state is enabled.")
        if udp_state_timeout <= 0:
            raise ValueError(f"udp_state_timeout must be > 0, got {udp_state_timeout}.")
        if tcp_offset_pose is not None:
            raise ValueError("tcp_offset_pose is not supported by RealmanInterpolationController.")
        if payload_mass is not None:
            raise ValueError("payload_mass is not supported by RealmanInterpolationController.")
        if payload_cog is not None:
            raise ValueError("payload_cog is not supported by RealmanInterpolationController.")
        if joints_init is not None:
            joints_init = np.array(joints_init)
            if joints_init.shape != (joint_dim,):
                raise ValueError(
                    f"joints_init shape must be ({joint_dim},), got {joints_init.shape}.")

        super().__init__(name="RealmanInterpolationController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.robot_level = level
        self.robot_mode = mode
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.joint_dim = joint_dim
        self.connect_retry_count = connect_retry_count
        self.connect_retry_interval = connect_retry_interval
        self.command_mode = command_mode
        self.interpolation_mode = interpolation_mode
        self.follow = follow
        self.trajectory_mode = trajectory_mode
        self.radio = radio
        self.move_speed = move_speed
        self.move_radius = move_radius
        self.move_connect = move_connect
        self.move_block = move_block
        self.state_read_retries = state_read_retries
        self.state_read_retry_delay = state_read_retry_delay
        self.max_consecutive_state_read_failures = max_consecutive_state_read_failures
        self.use_udp_state = use_udp_state
        self.udp_port = udp_port
        self.udp_cycle = udp_cycle
        self.udp_target_ip = udp_target_ip
        self.udp_state_timeout = udp_state_timeout

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        example = _make_receive_example(receive_keys, joint_dim)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.error_queue = mp.Queue()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RealmanInterpolationController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        deadline = time.monotonic() + self.launch_timeout
        while time.monotonic() < deadline:
            try:
                message = self.error_queue.get_nowait()
            except Exception:
                message = None
            if message is not None:
                raise RuntimeError(
                    f"RealmanInterpolationController process failed during startup:\n{message}")
            if self.ready_event.is_set():
                if not self.is_alive():
                    raise RuntimeError("RealmanInterpolationController process exited before becoming ready.")
                return
            if not self.is_alive():
                raise RuntimeError("RealmanInterpolationController process exited before becoming ready.")
            time.sleep(0.05)

        raise RuntimeError(
            f"RealmanInterpolationController did not become ready within "
            f"{self.launch_timeout:.1f}s.")
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        if not self.is_alive():
            raise RuntimeError("RealmanInterpolationController process is not alive.")
        if duration < (1/self.frequency):
            raise ValueError(
                f"duration must be >= control period {1/self.frequency}, got {duration}.")
        pose = np.array(pose)
        if pose.shape != (6,):
            raise ValueError(f"servoL pose shape must be (6,), got {pose.shape}.")

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        if target_time is None:
            raise ValueError("RealmanInterpolationController.schedule_waypoint requires target_time.")
        pose = np.array(pose)
        if pose.shape != (6,):
            raise ValueError(f"schedule_waypoint pose shape must be (6,), got {pose.shape}.")

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    def _connect_robot(self):
        thread_mode = rm_thread_mode_e(self.robot_mode)
        robot = RoboticArm(thread_mode)
        handle = None
        for attempt in range(self.connect_retry_count):
            handle = robot.rm_create_robot_arm(
                self.robot_ip, self.robot_port, self.robot_level)
            if handle.id != -1:
                return robot
            if attempt < self.connect_retry_count - 1:
                time.sleep(self.connect_retry_interval)
        raise RuntimeError(
            f"Failed to connect to Realman arm at {self.robot_ip}:{self.robot_port}")

    def _setup_udp_state(self, robot):
        self._udp_state_lock = threading.Lock()
        self._last_udp_state = None
        self._last_udp_state_time = None

        def callback(state_data):
            waypoint = state_data.waypoint
            pose = np.array([
                waypoint.position.x,
                waypoint.position.y,
                waypoint.position.z,
                waypoint.euler.rx,
                waypoint.euler.ry,
                waypoint.euler.rz,
            ], dtype=np.float64)
            joint_deg = np.asarray(state_data.joint_status.joint_position, dtype=np.float64)
            joint = np.deg2rad(joint_deg[:self.joint_dim])
            with self._udp_state_lock:
                self._last_udp_state = (pose, joint)
                self._last_udp_state_time = time.time()

        config = rm_realtime_push_config_t(
            cycle=self.udp_cycle,
            enable=True,
            port=self.udp_port,
            force_coordinate=-1,
            ip=self.udp_target_ip)
        ret = robot.rm_set_realtime_push(config)
        if ret != 0:
            raise RuntimeError(
                "Failed to enable Realman UDP realtime push, "
                f"error code: {ret}, target_ip={self.udp_target_ip}, "
                f"port={self.udp_port}, cycle={self.udp_cycle}")

        callback_type = CFUNCTYPE(None, rm_realtime_arm_joint_state_t)
        self._udp_callback = callback_type(callback)
        self._udp_callback_ref = self._udp_callback
        robot.rm_realtime_arm_state_call_back(self._udp_callback)

    def _get_udp_arm_state_once(self):
        with self._udp_state_lock:
            state = self._last_udp_state
            state_time = self._last_udp_state_time
        if state is None:
            return None

        pose, joint = state
        pose = _check_vector('udp_pose', pose, (6,))
        joint = _check_vector('udp_joint', joint, (self.joint_dim,))
        return pose, joint, state_time

    def _get_udp_arm_state(self):
        deadline = time.time() + self.udp_state_timeout
        while time.time() < deadline:
            result = self._get_udp_arm_state_once()
            if result is not None:
                pose, joint, state_time = result
                if state_time is not None and (time.time() - state_time) <= self.udp_state_timeout:
                    return pose, joint
            time.sleep(0.001)
        raise RuntimeError(
            f"Timed out waiting for Realman UDP state for {self.udp_state_timeout:.3f}s. "
            "Check robot_mode is triple-thread mode, UDP config, and firewall.")

    def _get_tcp_arm_state(self, robot):
        last_ret = None
        for attempt in range(self.state_read_retries):
            ret, raw_state = robot.rm_get_current_arm_state()
            if ret == 0:
                pose = _check_vector('pose', raw_state['pose'], (6,))
                joint = _check_vector('joint', raw_state['joint'], (self.joint_dim,))
                return pose, joint
            last_ret = ret
            if attempt < self.state_read_retries - 1 and self.state_read_retry_delay > 0:
                time.sleep(self.state_read_retry_delay)
        raise RuntimeError(
            "Failed to get Realman arm state after "
            f"{self.state_read_retries} attempts, last error code: {last_ret}")

    def _get_arm_state(self, robot):
        if self.use_udp_state:
            return self._get_udp_arm_state()
        return self._get_tcp_arm_state(robot)

    def _send_pose_command(self, robot, pose):
        pose = np.asarray(pose, dtype=np.float64).tolist()
        if self.command_mode == 'movep_canfd':
            return robot.rm_movep_canfd(
                pose,
                follow=self.follow,
                trajectory_mode=self.trajectory_mode,
                radio=self.radio)
        if self.command_mode == 'movep_follow':
            return robot.rm_movep_follow(pose)
        if self.command_mode == 'movej_p':
            return robot.rm_movej_p(
                pose,
                self.move_speed,
                self.move_radius,
                self.move_connect,
                self.move_block)
        return robot.rm_movel(
            pose,
            self.move_speed,
            self.move_radius,
            self.move_connect,
            self.move_block)

    def _build_ring_state(self,
            actual_pose,
            actual_joint,
            target_pose,
            prev_pose,
            prev_joint,
            prev_time,
            timestamp):
        tcp_speed = np.zeros((6,), dtype=np.float64)
        joint_vel = np.zeros((self.joint_dim,), dtype=np.float64)
        if prev_pose is not None and prev_joint is not None and prev_time is not None:
            dt = max(timestamp - prev_time, 1e-6)
            tcp_speed = (actual_pose - prev_pose) / dt
            joint_vel = (actual_joint - prev_joint) / dt

        state = dict()
        for key in self.receive_keys:
            if key == 'ActualTCPPose':
                state[key] = actual_pose
            elif key == 'ActualTCPSpeed':
                state[key] = tcp_speed
            elif key == 'ActualQ':
                state[key] = actual_joint
            elif key == 'ActualQd':
                state[key] = joint_vel
            elif key == 'TargetTCPPose':
                state[key] = target_pose
            elif key == 'TargetTCPSpeed':
                state[key] = np.zeros((6,), dtype=np.float64)
            elif key == 'TargetQ':
                state[key] = actual_joint
            elif key == 'TargetQd':
                state[key] = np.zeros((self.joint_dim,), dtype=np.float64)
            else:
                raise ValueError(f"Unsupported receive key for Realman controller: {key}")
        state['robot_receive_timestamp'] = timestamp
        state['robot_timestamp'] = timestamp - self.receive_latency
        return state
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        robot = None

        try:
            robot = self._connect_robot()
            if self.verbose:
                print(f"[RealmanInterpolationController] Connected to robot: {self.robot_ip}:{self.robot_port}")

            if self.use_udp_state:
                self._setup_udp_state(robot)
                if self.verbose:
                    print(
                        "[RealmanInterpolationController] UDP realtime state enabled: "
                        f"target_ip={self.udp_target_ip}, port={self.udp_port}, cycle={self.udp_cycle}")

            if self.joints_init is not None:
                ret = robot.rm_movej(
                    self.joints_init.tolist(),
                    int(self.joints_init_speed),
                    self.move_radius,
                    self.move_connect,
                    1)
                if ret != 0:
                    raise RuntimeError(f"Realman moveJ init failed with error code: {ret}")

            # main loop
            dt = 1. / self.frequency
            curr_pose, curr_joint = self._get_arm_state(robot)
            curr_interp_pose = _realman_euler_pose_to_rotvec_pose(curr_pose)
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_interp_pose]
            )
            direct_target_pose = curr_pose.copy()
            pending_direct_waypoints = list()
            
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            prev_pose = None
            prev_joint = None
            prev_timestamp = None
            consecutive_state_read_failures = 0
            while keep_running:
                # start control iteration
                t_now = time.monotonic()
                if self.interpolation_mode == 'none':
                    while pending_direct_waypoints and pending_direct_waypoints[0][0] <= t_now:
                        _, direct_target_pose = pending_direct_waypoints.pop(0)
                    pose_command = direct_target_pose
                else:
                    pose_command_interp = pose_interp(t_now)
                    pose_command = _rotvec_pose_to_realman_euler_pose(pose_command_interp)
                ret = self._send_pose_command(robot, pose_command)
                if ret != 0:
                    raise RuntimeError(f"Realman pose command failed with error code: {ret}")
                
                # update robot state
                t_recv = time.time()
                try:
                    actual_pose, actual_joint = self._get_arm_state(robot)
                except RuntimeError as e:
                    consecutive_state_read_failures += 1
                    if self.verbose:
                        print(
                            "[RealmanInterpolationController] "
                            f"state read failed {consecutive_state_read_failures}/"
                            f"{self.max_consecutive_state_read_failures}: {e}")
                    if consecutive_state_read_failures >= self.max_consecutive_state_read_failures:
                        raise RuntimeError(
                            "Realman arm state read failed consecutively "
                            f"{consecutive_state_read_failures} times.") from e
                else:
                    consecutive_state_read_failures = 0
                    state = self._build_ring_state(
                        actual_pose=actual_pose,
                        actual_joint=actual_joint,
                        target_pose=pose_command,
                        prev_pose=prev_pose,
                        prev_joint=prev_joint,
                        prev_time=prev_timestamp,
                        timestamp=t_recv)
                    self.ring_buffer.put(state)
                    prev_pose = actual_pose
                    prev_joint = actual_joint
                    prev_timestamp = t_recv

                # fetch commands from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        if self.interpolation_mode == 'none':
                            pending_direct_waypoints.append((t_insert, target_pose))
                            pending_direct_waypoints.sort(key=lambda x: x[0])
                        else:
                            target_interp_pose = _realman_euler_pose_to_rotvec_pose(target_pose)
                            pose_interp = pose_interp.drive_to_waypoint(
                                pose=target_interp_pose,
                                time=t_insert,
                                curr_time=curr_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed
                            )
                            last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RealmanInterpolationController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        if self.interpolation_mode == 'none':
                            if target_time <= curr_time:
                                direct_target_pose = target_pose
                            else:
                                pending_direct_waypoints.append((target_time, target_pose))
                                pending_direct_waypoints.sort(key=lambda x: x[0])
                        else:
                            target_interp_pose = _realman_euler_pose_to_rotvec_pose(target_pose)
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_interp_pose,
                                time=target_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[RealmanInterpolationController] Actual frequency {1/(time.monotonic() - t_now)}")

        except BaseException:
            self.error_queue.put(traceback.format_exc())
            self.ready_event.set()
            raise
        finally:
            if robot is not None:
                robot.rm_delete_robot_arm()

            if self.verbose:
                print(f"[RealmanInterpolationController] Disconnected from robot: {self.robot_ip}")
