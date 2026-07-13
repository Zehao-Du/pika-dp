import time
import enum
import logging
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np

try:
    from umi.shared_memory.shared_memory_queue import (
        SharedMemoryQueue, Empty)
    from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
except ImportError as e:
    raise ImportError(
        "Missing dependency for UMI shared memory. Install required packages "
        "in the active Python environment, for example: uv pip install atomics"
    ) from e
from umi.common.precise_sleep import precise_wait

try:
    from pika.gripper import Gripper
except ImportError as e:
    raise ImportError(
        "Missing dependency: pika SDK is required for PikaController. "
        "Install it in the active Python environment, for example: "
        "uv pip install -e /home/ubuntu/Documents/CodeField/zehao/pika_sdk --no-deps"
    ) from e

logging.getLogger("pika.gripper").setLevel(logging.WARNING)

DEFAULT_GRIPPER_VELOCITY = 0.1
DEFAULT_GRIPPER_WIDTH_MIN = 0.0
DEFAULT_GRIPPER_WIDTH_MAX = 90.0
DEFAULT_GRIPPER_SERIAL_PORT = "/dev/ttyUSB60"


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class PikaController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            frequency=30,
            move_max_speed=200.0,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            use_meters=False,
            min_width=DEFAULT_GRIPPER_WIDTH_MIN,
            max_width=DEFAULT_GRIPPER_WIDTH_MAX,
            init_velocity=DEFAULT_GRIPPER_VELOCITY,
            serial_port=DEFAULT_GRIPPER_SERIAL_PORT,
            verbose=False
            ):
        if shm_manager is None:
            raise ValueError("PikaController requires shm_manager.")
        if frequency <= 0:
            raise ValueError(f"PikaController frequency must be > 0, got {frequency}.")
        if move_max_speed <= 0:
            raise ValueError(f"PikaController move_max_speed must be > 0, got {move_max_speed}.")
        if command_queue_size <= 0:
            raise ValueError(f"PikaController command_queue_size must be > 0, got {command_queue_size}.")
        if launch_timeout <= 0:
            raise ValueError(f"PikaController launch_timeout must be > 0, got {launch_timeout}.")
        if min_width >= max_width:
            raise ValueError(
                f"PikaController min_width must be smaller than max_width, got {min_width} >= {max_width}.")
        if not serial_port:
            raise ValueError("PikaController requires serial_port.")

        super().__init__(name="PikaController")
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.min_width = min_width
        self.max_width = max_width
        self.init_velocity = init_velocity
        self.serial_port = serial_port
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PikaController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        if self.ready_event.wait(self.launch_timeout):
            if not self.is_alive():
                raise RuntimeError("PikaController process exited before becoming ready.")
            return
        if not self.is_alive():
            raise RuntimeError("PikaController process exited before becoming ready.")
        raise RuntimeError(
            f"PikaController did not become ready within {self.launch_timeout:.1f}s. "
            f"serial_port={self.serial_port}")
    
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
    def schedule_waypoint(self, pos: float, target_time: float):
        if pos is None:
            raise ValueError("PikaController.schedule_waypoint requires pos.")
        if target_time is None:
            raise ValueError("PikaController.schedule_waypoint requires target_time.")
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)


    def restart_put(self, start_time):
        if start_time is None:
            raise ValueError("PikaController.restart_put requires start_time.")
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def _connect_gripper(self):
        gripper = Gripper(port=self.serial_port)
        if not gripper.connect():
            raise RuntimeError(f"Failed to connect to the Pika gripper at {self.serial_port}")
        if not gripper.enable():
            raise RuntimeError("Failed to enable the Pika gripper")
        if self.init_velocity is not None:
            gripper.set_velocity(self.init_velocity)
        return gripper

    def run(self):
        # start connection
        gripper = None
        try:
            gripper = self._connect_gripper()

            curr_pos = float(gripper.get_gripper_distance())
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            prev_pos = curr_pos
            prev_time = time.time()
            pending_target_pos = None
            pending_target_time = None
            while keep_running:
                t_now = time.monotonic()
                if pending_target_pos is not None and t_now >= pending_target_time:
                    gripper.set_gripper_distance(pending_target_pos)
                    pending_target_pos = None
                    pending_target_time = None

                # get state from gripper
                receive_time = time.time()
                curr_pos = float(gripper.get_gripper_distance())
                state_dt = max(receive_time - prev_time, 1e-6)
                curr_vel = (curr_pos - prev_pos) / state_dt
                state = {
                    'gripper_state': 0,
                    'gripper_position': curr_pos / self.scale,
                    'gripper_velocity': curr_vel / self.scale,
                    'gripper_force': 0.0,
                    'gripper_measure_timestamp': receive_time,
                    'gripper_receive_timestamp': receive_time,
                    'gripper_timestamp': receive_time - self.receive_latency
                }
                self.ring_buffer.put(state)
                prev_pos = curr_pos
                prev_time = receive_time

                # fetch command from queue
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
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos'] * self.scale
                        target_pos = float(np.clip(target_pos, self.min_width, self.max_width))
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        pending_target_pos = target_pos
                        pending_target_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break
                    
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
                
        finally:
            self.ready_event.set()
            if self.verbose:
                print("[PikaController] Disconnected from gripper")
