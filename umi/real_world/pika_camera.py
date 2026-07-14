from typing import Callable, Dict, List, Optional, Union
import enum
import pathlib
import time
import traceback
import multiprocessing as mp

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "pika_camera requires OpenCV. Install cv2/opencv-python in this environment."
    ) from e

import numpy as np
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager

from umi.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from umi.shared_memory.shared_ndarray import SharedNDArray
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.real_world.video_recorder import VideoRecorder


DEFAULT_FISHEYE_DEVICE = "/dev/video60" #"/dev/v4l/by-path/pci-0000:00:14.0-usb-0:4.1:1.0-video-index0"
DEFAULT_REALSENSE_SERIAL = "419122270755"
DEFAULT_RESOLUTION = (640, 480)
DEFAULT_FPS = 30


class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


def _import_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise ImportError(
            "PikaCamera requires pyrealsense2 for the RealSense RGBD camera. "
            "Install pyrealsense2 in the active environment."
        ) from e
    return rs


class PikaCamera(mp.Process):
    """
    Process-backed camera wrapper for the Pika setup.

    It reads:
      - fisheye RGB from an OpenCV/V4L2 camera
      - RGB + depth from an Intel RealSense camera

    Ring-buffer samples use direct observation keys:
      fisheye: RGB uint8 or transformed output
      rgb: RGB uint8 or transformed output
      depth: metric depth float32, shaped H,W,1 after env transform
    """

    MAX_PATH_LENGTH = 4096

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        fisheye_device: str,
        realsense_serial: str,
        resolution=DEFAULT_RESOLUTION,
        capture_fps=DEFAULT_FPS,
        put_fps=None,
        put_downsample=True,
        get_max_k=30,
        receive_latency=0.0,
        cap_buffer_size=1,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[Union[List[VideoRecorder], tuple]] = None,
        launch_timeout=10.0,
        num_threads=2,
        verbose=False,
    ):
        super().__init__()

        if shm_manager is None:
            raise ValueError("PikaCamera requires shm_manager.")
        if not fisheye_device:
            raise ValueError("PikaCamera requires fisheye_device.")
        if not realsense_serial:
            raise ValueError("PikaCamera requires realsense_serial.")
        if resolution is None or len(resolution) != 2:
            raise ValueError(f"PikaCamera requires resolution=(width,height), got {resolution}.")
        if capture_fps is None or capture_fps <= 0:
            raise ValueError(f"PikaCamera requires capture_fps > 0, got {capture_fps}.")
        if launch_timeout is None or launch_timeout <= 0:
            raise ValueError(f"PikaCamera requires launch_timeout > 0, got {launch_timeout}.")

        # Fail early for missing RealSense dependency instead of hanging during start().
        _import_realsense()

        if put_fps is None:
            put_fps = capture_fps

        resolution = tuple(int(x) for x in resolution)
        shape = resolution[::-1]
        examples = {
            "fisheye": np.empty(shape=shape + (3,), dtype=np.uint8),
            "rgb": np.empty(shape=shape + (3,), dtype=np.uint8),
            "depth": np.empty(shape=shape, dtype=np.float32),
            "camera_capture_timestamp": 0.0,
            "camera_receive_timestamp": 0.0,
            "timestamp": 0.0,
            "step_idx": 0,
        }

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        vis_examples = {
            "color": np.empty(shape=(2,) + shape + (3,), dtype=np.uint8),
            "timestamp": 0.0,
        }
        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=vis_examples if vis_transform is None else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )

        command_examples = {
            "cmd": Command.RESTART_PUT.value,
            "put_start_time": 0.0,
            "fisheye_video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "rgb_video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
        }
        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=command_examples,
            buffer_size=128,
        )
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager,
            shape=(3, 3),
            dtype=np.float64,
        )
        intrinsics_array.get()[:] = np.eye(3, dtype=np.float64)

        if video_recorder is None:
            video_recorder = [
                VideoRecorder.create_h264(
                    fps=capture_fps,
                    input_pix_fmt="rgb24",
                ),
                VideoRecorder.create_h264(
                    fps=capture_fps,
                    input_pix_fmt="rgb24",
                ),
            ]
        if len(video_recorder) != 2:
            raise ValueError("PikaCamera video_recorder must contain two recorders: fisheye and rgb.")
        for recorder in video_recorder:
            if recorder.fps != capture_fps:
                raise ValueError(
                    f"PikaCamera recorder fps {recorder.fps} does not match capture_fps {capture_fps}."
                )

        self.shm_manager = shm_manager
        self.fisheye_device = fisheye_device
        self.realsense_serial = realsense_serial
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.get_max_k = get_max_k
        self.receive_latency = receive_latency
        self.cap_buffer_size = cap_buffer_size
        self.transform = transform
        self.vis_transform = vis_transform
        self.video_recorder = list(video_recorder)
        self.launch_timeout = launch_timeout
        self.num_threads = num_threads
        self.verbose = verbose
        self.put_start_time = None

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
        self.error_queue = mp.Queue()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        # Recorded RGB streams: fisheye and RealSense color.
        return 2

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        data_example = np.empty(self.resolution[::-1] + (3,), dtype=np.uint8)
        for recorder in self.video_recorder:
            recorder.start(
                shm_manager=self.shm_manager,
                data_example=data_example,
            )
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for recorder in self.video_recorder:
            recorder.stop()
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        if not self.ready_event.wait(self.launch_timeout):
            message = self._pop_error_message()
            if message is not None:
                raise RuntimeError(f"PikaCamera failed to start:\n{message}")
            if self.exitcode is not None:
                raise RuntimeError(f"PikaCamera exited before becoming ready. exitcode={self.exitcode}")
            raise RuntimeError(
                f"PikaCamera did not become ready within {self.launch_timeout:.1f}s. "
                f"fisheye_device={self.fisheye_device}, realsense_serial={self.realsense_serial}"
            )
        for recorder in self.video_recorder:
            recorder.start_wait()

    def stop_wait(self):
        self.join()
        for recorder in self.video_recorder:
            recorder.end_wait()

    def end_wait(self):
        self.stop_wait()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def get_intrinsics(self):
        if not self.ready_event.is_set():
            raise RuntimeError("PikaCamera must be ready before reading intrinsics.")
        return self.intrinsics_array.get().copy()

    def restart_put(self, start_time):
        if start_time is None:
            raise ValueError("PikaCamera.restart_put requires start_time.")
        self.command_queue.put({
            "cmd": Command.RESTART_PUT.value,
            "put_start_time": start_time,
        })

    def start_recording(self, video_path: Union[str, List[str]], start_time: float = -1):
        paths = self._resolve_video_paths(video_path)
        self.command_queue.put({
            "cmd": Command.START_RECORDING.value,
            "fisheye_video_path": paths[0],
            "rgb_video_path": paths[1],
            "recording_start_time": start_time,
        })

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def _resolve_video_paths(self, video_path: Union[str, List[str]]) -> List[str]:
        if isinstance(video_path, str):
            video_dir = pathlib.Path(video_path)
            if not video_dir.parent.is_dir():
                raise ValueError(f"Video directory parent does not exist: {video_dir.parent}")
            video_dir.mkdir(parents=True, exist_ok=True)
            paths = [
                str(video_dir.joinpath("0.mp4").absolute()),
                str(video_dir.joinpath("1.mp4").absolute()),
            ]
        else:
            paths = list(video_path)
        if len(paths) != 2:
            raise ValueError(f"PikaCamera.start_recording expects two video paths, got {len(paths)}.")
        for path in paths:
            path_len = len(path.encode("utf-8"))
            if path_len > self.MAX_PATH_LENGTH:
                raise RuntimeError(f"video_path too long: {path}")
        return paths

    def _pop_error_message(self):
        try:
            return self.error_queue.get_nowait()
        except Exception:
            return None

    def _open_fisheye(self):
        cap = cv2.VideoCapture(self.fisheye_device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open fisheye camera device {self.fisheye_device}.")

        width, height = self.resolution
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffer_size)
        return cap

    def _open_realsense(self, rs):
        width, height = self.resolution
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.realsense_serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.capture_fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.capture_fps)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.intrinsics_array.get()[:] = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        for _ in range(10):
            pipeline.wait_for_frames()
        return pipeline, align, depth_scale

    def _read_fisheye_rgb(self, cap):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame from fisheye camera {self.fisheye_device}.")
        expected_shape = self.resolution[::-1] + (3,)
        if frame.shape != expected_shape:
            raise RuntimeError(
                f"Fisheye frame shape {frame.shape} does not match expected {expected_shape}. "
                f"Check device resolution/fps settings."
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _read_realsense_rgbd(self, rs, pipeline, align, depth_scale):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None:
            raise RuntimeError(f"Failed to read color frame from RealSense {self.realsense_serial}.")
        if depth_frame is None:
            raise RuntimeError(f"Failed to read depth frame from RealSense {self.realsense_serial}.")

        color = np.asarray(color_frame.get_data())
        depth_raw = np.asarray(depth_frame.get_data())
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth_m = depth_raw.astype(np.float32) * float(depth_scale)
        capture_timestamp = color_frame.get_timestamp() / 1000.0
        return rgb, depth_m, capture_timestamp

    def run(self):
        try:
            self._run()
        except BaseException:
            self.error_queue.put(traceback.format_exc())
            raise

    def _run(self):
        threadpool_limits(self.num_threads)
        cv2.setNumThreads(self.num_threads)
        rs = _import_realsense()

        cap = None
        pipeline = None
        try:
            cap = self._open_fisheye()
            pipeline, align, depth_scale = self._open_realsense(rs)

            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                fisheye_rgb = self._read_fisheye_rgb(cap)
                rgb, depth_m, rs_capture_timestamp = self._read_realsense_rgbd(
                    rs, pipeline, align, depth_scale)
                receive_time = time.time()
                timestamp = receive_time - self.receive_latency

                data = {
                    "fisheye": fisheye_rgb,
                    "rgb": rgb,
                    "depth": depth_m,
                    "camera_receive_timestamp": receive_time,
                    "camera_capture_timestamp": rs_capture_timestamp,
                }

                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:
                    _, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[timestamp],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        put_data["timestamp"] = timestamp
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((timestamp - put_start_time) * self.put_fps)
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = timestamp
                    self.ring_buffer.put(put_data, wait=False)

                if iter_idx == 0:
                    self.ready_event.set()

                if self.video_recorder[0].is_ready():
                    self.video_recorder[0].write_frame(fisheye_rgb, frame_time=timestamp)
                if self.video_recorder[1].is_ready():
                    self.video_recorder[1].write_frame(rgb, frame_time=timestamp)

                if self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                else:
                    vis_data = {
                        "color": np.stack([fisheye_rgb, rgb], axis=0),
                        "timestamp": timestamp,
                    }
                self.vis_ring_buffer.put(vis_data, wait=False)

                if self.verbose:
                    t_end = time.time()
                    print(f"[PikaCamera] FPS {np.round(1 / (t_end - t_start), 1)}")
                    t_start = t_end

                put_start_time, put_idx = self._handle_commands(put_start_time, put_idx)
                iter_idx += 1

        finally:
            for recorder in self.video_recorder:
                recorder.stop()
            if pipeline is not None:
                pipeline.stop()
            if cap is not None:
                cap.release()

    def _handle_commands(self, put_start_time, put_idx):
        try:
            commands = self.command_queue.get_all()
            n_cmd = len(commands["cmd"])
        except Empty:
            return put_start_time, put_idx

        for i in range(n_cmd):
            cmd = commands["cmd"][i]
            if cmd == Command.RESTART_PUT.value:
                put_start_time = commands["put_start_time"][i]
                put_idx = None
            elif cmd == Command.START_RECORDING.value:
                start_time = commands["recording_start_time"][i]
                if start_time < 0:
                    start_time = None
                self.video_recorder[0].start_recording(
                    str(commands["fisheye_video_path"][i]),
                    start_time=start_time,
                )
                self.video_recorder[1].start_recording(
                    str(commands["rgb_video_path"][i]),
                    start_time=start_time,
                )
            elif cmd == Command.STOP_RECORDING.value:
                for recorder in self.video_recorder:
                    recorder.stop_recording()
                put_idx = None
            else:
                raise RuntimeError(f"Unknown PikaCamera command: {cmd}")
        return put_start_time, put_idx
