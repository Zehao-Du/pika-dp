"""JSON-driven real-world Map4D construction.

Each task is defined by ``representation/realworld/<task_name>.json`` with the
task name as its top-level key. Required object schema:

.. code-block:: json

    {
      "TaskName": {
        "size_parameters": {"dim": 3, "default": [0.1, 0.2, 0.3]},
        "objects": {
          "object_name": {
            "sam3": {
              "prompt": "the object",
              "selection": "highest_score",
              "min_score": 0.5
            },
            "foundationpose": {
              "mesh_path": "meshes/object.obj",
              "refine_iterations": 3
            },
            "size_parameters": [
              {
                "name": "height",
                "primitive": "body",
                "global_index": 0,
                "default": 0.1
              }
            ],
            "primitives": [
              {
                "name": "body",
                "type": "Cuboid",
                "semantic": "object body",
                "parameters": {"height": "height", "top_length": 0.2, "top_width": 0.3},
                "arguments": {},
                "local_pose": {
                  "position": [0, 0, 0],
                  "rotation_6d": [1, 0, 0, 0, 1, 0]
                }
              }
            ]
          }
        }
      }
    }

SAM3 produces one selected mask per object. FoundationPose registers the
object mesh from that mask and produces ``camera_from_object``. Primitive local
poses are then composed with the object pose to instantiate the scene Map4D.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image
import torch


REALWORLD_DIR = Path(__file__).resolve().parent / "representation" / "realworld"
IDENTITY_ROTATION_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
FOUNDATIONPOSE_MIN_VALID_DEPTH_PIXELS = 4
_RESERVED_PRIMITIVE_ARGUMENTS = {
    "position",
    "rotation",
    "Semantic",
    "Affordance",
}


@dataclass(frozen=True)
class SizeParameterSpec:
    name: str
    primitive: str
    global_index: int
    default: float


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    primitive_type: str
    semantic: str
    parameters: Mapping[str, Any]
    arguments: Mapping[str, Any]
    local_position: tuple[float, float, float]
    local_rotation_6d: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    prompt: str
    selection: str
    instance_index: Optional[int]
    min_score: float
    mesh_path: Path
    refine_iterations: int
    size_parameters: tuple[SizeParameterSpec, ...]
    primitives: tuple[PrimitiveSpec, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    metadata_path: Path
    size_parameters: tuple[float, ...]
    objects: tuple[ObjectSpec, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    box_xyxy: np.ndarray
    score: float
    instance_index: int


@dataclass(frozen=True)
class ObjectPoseResult:
    object_spec: ObjectSpec
    segmentation: SegmentationResult
    camera_from_object: np.ndarray
    output_from_object: np.ndarray


@dataclass(frozen=True)
class ConstructionResult:
    task_spec: TaskSpec
    map4d: Any
    size_parameters: torch.Tensor
    positions: torch.Tensor
    rotations_6d: torch.Tensor
    objects: tuple[ObjectPoseResult, ...]


def _require_mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_sequence(value, name, length=None):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a JSON array")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(value)}")
    return value


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(value, name, *, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _finite_tuple(value, name, length):
    values = tuple(
        _finite_float(item, f"{name}[{index}]")
        for index, item in enumerate(_require_sequence(value, name, length))
    )
    return values


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _resolve_mesh_path(value, metadata_path, name):
    raw_path = Path(_nonempty_string(value, name)).expanduser()
    path = raw_path if raw_path.is_absolute() else metadata_path.parent / raw_path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _load_primitive_spec(
    raw,
    object_name,
    parameter_names_by_primitive,
    index,
):
    raw = _require_mapping(raw, f"objects.{object_name}.primitives[{index}]")
    prefix = f"objects.{object_name}.primitives[{index}]"
    name = _nonempty_string(raw.get("name"), f"{prefix}.name")
    allowed_parameter_names = parameter_names_by_primitive.get(name, set())
    primitive_type = _nonempty_string(raw.get("type"), f"{prefix}.type")
    semantic = _nonempty_string(raw.get("semantic"), f"{prefix}.semantic")
    parameters = dict(_require_mapping(raw.get("parameters"), f"{prefix}.parameters"))
    if not parameters:
        raise ValueError(f"{prefix}.parameters must not be empty")
    reserved_parameters = _RESERVED_PRIMITIVE_ARGUMENTS.intersection(parameters)
    if reserved_parameters:
        raise ValueError(
            f"{prefix}.parameters contains reserved keys: "
            f"{sorted(reserved_parameters)}"
        )
    for parameter_name, source in parameters.items():
        _nonempty_string(parameter_name, f"{prefix}.parameters key")
        if isinstance(source, str):
            if source not in allowed_parameter_names:
                raise KeyError(
                    f"{prefix}.parameters.{parameter_name} references size "
                    f"parameter {source!r} not assigned to primitive {name!r}"
                )
        elif isinstance(source, (int, float)) and not isinstance(source, bool):
            _finite_float(source, f"{prefix}.parameters.{parameter_name}")
        else:
            raise ValueError(
                f"{prefix}.parameters.{parameter_name} must be a size parameter "
                "name or numeric constant"
            )

    arguments = dict(
        _require_mapping(raw.get("arguments"), f"{prefix}.arguments")
    )
    conflicts = _RESERVED_PRIMITIVE_ARGUMENTS.intersection(arguments)
    if conflicts:
        raise ValueError(
            f"{prefix}.arguments contains reserved keys: {sorted(conflicts)}"
        )
    duplicate_keywords = set(parameters).intersection(arguments)
    if duplicate_keywords:
        raise ValueError(
            f"{prefix}.parameters and arguments contain duplicate keys: "
            f"{sorted(duplicate_keywords)}"
        )
    local_pose = _require_mapping(raw.get("local_pose"), f"{prefix}.local_pose")
    local_position = _finite_tuple(
        local_pose.get("position"),
        f"{prefix}.local_pose.position",
        3,
    )
    local_rotation = _finite_tuple(
        local_pose.get("rotation_6d"),
        f"{prefix}.local_pose.rotation_6d",
        6,
    )
    _rotation_6d_to_matrix(np.asarray(local_rotation, dtype=np.float64))
    return PrimitiveSpec(
        name=name,
        primitive_type=primitive_type,
        semantic=semantic,
        parameters=parameters,
        arguments=arguments,
        local_position=local_position,
        local_rotation_6d=local_rotation,
    )


def _load_object_spec(name, raw, metadata_path):
    name = _nonempty_string(name, "object name")
    raw = _require_mapping(raw, f"objects.{name}")
    sam3 = _require_mapping(raw.get("sam3"), f"objects.{name}.sam3")
    prompt = _nonempty_string(sam3.get("prompt"), f"objects.{name}.sam3.prompt")
    selection = _nonempty_string(
        sam3.get("selection"),
        f"objects.{name}.sam3.selection",
    )
    if selection not in {"highest_score", "index"}:
        raise ValueError(
            f"objects.{name}.sam3.selection must be 'highest_score' or 'index'"
        )
    instance_index = sam3.get("index")
    if selection == "index":
        instance_index = _strict_int(
            instance_index,
            f"objects.{name}.sam3.index",
            minimum=0,
        )
    elif instance_index is not None:
        raise ValueError(
            f"objects.{name}.sam3.index is only valid when selection='index'"
        )
    min_score = _finite_float(
        sam3.get("min_score"),
        f"objects.{name}.sam3.min_score",
    )
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"objects.{name}.sam3.min_score must be in [0, 1]")

    foundationpose = _require_mapping(
        raw.get("foundationpose"),
        f"objects.{name}.foundationpose",
    )
    mesh_path = _resolve_mesh_path(
        foundationpose.get("mesh_path"),
        metadata_path,
        f"objects.{name}.foundationpose.mesh_path",
    )
    refine_iterations = _strict_int(
        foundationpose.get("refine_iterations"),
        f"objects.{name}.foundationpose.refine_iterations",
        minimum=1,
    )

    raw_size_parameters = _require_sequence(
        raw.get("size_parameters"),
        f"objects.{name}.size_parameters",
    )
    if not raw_size_parameters:
        raise ValueError(f"objects.{name}.size_parameters must not be empty")
    size_parameters = []
    parameter_names = set()
    for index, item in enumerate(raw_size_parameters):
        item = _require_mapping(item, f"objects.{name}.size_parameters[{index}]")
        prefix = f"objects.{name}.size_parameters[{index}]"
        parameter_name = _nonempty_string(item.get("name"), f"{prefix}.name")
        if parameter_name in parameter_names:
            raise ValueError(
                f"objects.{name} has duplicate size parameter {parameter_name!r}"
            )
        parameter_names.add(parameter_name)
        primitive_name = _nonempty_string(
            item.get("primitive"),
            f"{prefix}.primitive",
        )
        global_index = _strict_int(
            item.get("global_index"),
            f"{prefix}.global_index",
            minimum=0,
        )
        default = _finite_float(item.get("default"), f"{prefix}.default")
        if default <= 0.0:
            raise ValueError(f"{prefix}.default must be positive")
        size_parameters.append(
            SizeParameterSpec(
                name=parameter_name,
                primitive=primitive_name,
                global_index=global_index,
                default=default,
            )
        )

    raw_primitives = _require_sequence(
        raw.get("primitives"),
        f"objects.{name}.primitives",
    )
    if not raw_primitives:
        raise ValueError(f"objects.{name}.primitives must not be empty")
    parameter_names_by_primitive = {}
    for parameter in size_parameters:
        parameter_names_by_primitive.setdefault(parameter.primitive, set()).add(
            parameter.name
        )
    primitives = tuple(
        _load_primitive_spec(
            item,
            name,
            parameter_names_by_primitive,
            index,
        )
        for index, item in enumerate(raw_primitives)
    )
    primitive_names = [primitive.name for primitive in primitives]
    if len(primitive_names) != len(set(primitive_names)):
        raise ValueError(f"objects.{name} has duplicate primitive names")
    declared_primitive_names = {item.primitive for item in size_parameters}
    unknown_primitives = declared_primitive_names.difference(primitive_names)
    if unknown_primitives:
        raise ValueError(
            f"objects.{name}.size_parameters reference undeclared primitives: "
            f"{sorted(unknown_primitives)}"
        )
    for primitive in primitives:
        declared = parameter_names_by_primitive.get(primitive.name, set())
        referenced = {
            source
            for source in primitive.parameters.values()
            if isinstance(source, str)
        }
        if referenced != declared:
            missing = sorted(declared.difference(referenced))
            raise ValueError(
                f"objects.{name} primitive {primitive.name!r} does not consume "
                f"all assigned size parameters; missing {missing}"
            )
    return ObjectSpec(
        name=name,
        prompt=prompt,
        selection=selection,
        instance_index=instance_index,
        min_score=min_score,
        mesh_path=mesh_path,
        refine_iterations=refine_iterations,
        size_parameters=tuple(size_parameters),
        primitives=primitives,
        raw=raw,
    )


def load_realworld_task(
    task_name,
    *,
    metadata_root=REALWORLD_DIR,
    metadata_path=None,
):
    """Load and strictly validate one real-world task specification."""
    task_name = _nonempty_string(task_name, "task_name")
    if metadata_path is None:
        metadata_path = Path(metadata_root) / f"{task_name}.json"
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Real-world task metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload = _require_mapping(payload, str(metadata_path))
    if set(payload) != {task_name}:
        raise ValueError(
            f"{metadata_path} must contain exactly the top-level key {task_name!r}"
        )
    raw = _require_mapping(payload[task_name], task_name)
    size_meta = _require_mapping(
        raw.get("size_parameters"),
        f"{task_name}.size_parameters",
    )
    size_dim = _strict_int(
        size_meta.get("dim"),
        f"{task_name}.size_parameters.dim",
        minimum=1,
    )
    size_defaults = _finite_tuple(
        size_meta.get("default"),
        f"{task_name}.size_parameters.default",
        size_dim,
    )
    if any(value <= 0.0 for value in size_defaults):
        raise ValueError(f"{task_name}.size_parameters.default must be positive")
    raw_objects = _require_mapping(raw.get("objects"), f"{task_name}.objects")
    if not raw_objects:
        raise ValueError(f"{task_name}.objects must not be empty")
    objects = tuple(
        _load_object_spec(name, object_raw, metadata_path)
        for name, object_raw in raw_objects.items()
    )
    normalized_object_names = [object_spec.name for object_spec in objects]
    if len(normalized_object_names) != len(set(normalized_object_names)):
        raise ValueError(
            f"{task_name}.objects contains duplicate names after whitespace "
            "normalization"
        )

    indexed_parameters = {}
    for object_spec in objects:
        for parameter in object_spec.size_parameters:
            if parameter.global_index in indexed_parameters:
                previous = indexed_parameters[parameter.global_index]
                raise ValueError(
                    f"Duplicate global size parameter index {parameter.global_index}: "
                    f"{previous} and {object_spec.name}.{parameter.name}"
                )
            indexed_parameters[parameter.global_index] = (
                f"{object_spec.name}.{parameter.name}"
            )
            if parameter.global_index >= size_dim:
                raise ValueError(
                    f"{object_spec.name}.{parameter.name} global_index "
                    f"{parameter.global_index} exceeds size dim {size_dim}"
                )
            expected = size_defaults[parameter.global_index]
            if not np.isclose(parameter.default, expected):
                raise ValueError(
                    f"{object_spec.name}.{parameter.name} default "
                    f"{parameter.default} != task default {expected}"
                )
    expected_indices = set(range(size_dim))
    if set(indexed_parameters) != expected_indices:
        missing = sorted(expected_indices.difference(indexed_parameters))
        raise ValueError(f"Task size parameter indices are incomplete; missing {missing}")
    return TaskSpec(
        name=task_name,
        metadata_path=metadata_path,
        size_parameters=size_defaults,
        objects=objects,
        raw=raw,
    )


def build_sam3_processor(
    *,
    device="cuda",
    checkpoint_path=None,
    load_from_hf=True,
):
    """Build the official Meta SAM3 image processor."""
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if not isinstance(load_from_hf, bool):
        raise ValueError("load_from_hf must be a boolean")
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=bool(load_from_hf),
    )
    return Sam3Processor(
        model,
        device=device,
        confidence_threshold=0.0,
    )


def _as_rgb_uint8(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape [H, W, 3], got {rgb.shape}")
    if not np.issubdtype(rgb.dtype, np.number):
        raise ValueError("rgb must have a numeric dtype")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("rgb must contain only finite values")
    rgb_min = float(rgb.min())
    rgb_max = float(rgb.max())
    if rgb_min < 0.0:
        raise ValueError("rgb values must be non-negative")
    if np.issubdtype(rgb.dtype, np.floating):
        if rgb_max <= 1.0:
            rgb = rgb * 255.0
        elif rgb_max > 255.0:
            raise ValueError("floating-point rgb values must be in [0,1] or [0,255]")
    elif rgb_max > 255.0:
        raise ValueError("integer rgb values must be in [0,255]")
    return rgb.astype(np.uint8)


def _as_depth_float32(depth, image_shape):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != image_shape[:2]:
        raise ValueError(
            f"depth shape {depth.shape} must match rgb shape {image_shape[:2]}"
        )
    if not np.all(np.isfinite(depth)):
        raise ValueError("depth must contain only finite values")
    if np.any(depth < 0.0):
        raise ValueError("depth values must be non-negative")
    return depth


def _as_intrinsics(intrinsics):
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("camera_intrinsics must be a finite [3, 3] matrix")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("camera_intrinsics focal lengths must be positive")
    return intrinsics


def _as_transform(transform, name):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite [4, 4] matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} must be a homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError(f"{name} rotation must be orthonormal")
    if np.linalg.det(rotation) < 0.999 or np.linalg.det(rotation) > 1.001:
        raise ValueError(f"{name} rotation determinant must be 1")
    return transform


def _rotation_6d_to_matrix(rotation_6d):
    rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
    if rotation_6d.shape != (6,) or not np.all(np.isfinite(rotation_6d)):
        raise ValueError("rotation_6d must contain six finite values")
    first = rotation_6d[:3]
    second = rotation_6d[3:6]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        raise ValueError("rotation_6d first axis must be non-zero")
    first = first / first_norm
    second = second - np.dot(first, second) * first
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        raise ValueError("rotation_6d axes must be linearly independent")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=0)


def _matrix_to_rotation_6d(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    return np.concatenate([rotation[0, :], rotation[1, :]])


def _pose_from_local(position, rotation_6d):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_6d_to_matrix(rotation_6d)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


class SAM3Segmenter:
    """Strict adapter around the official SAM3 image processor."""

    def __init__(self, processor):
        for method in (
            "set_image",
            "set_text_prompt",
            "set_confidence_threshold",
        ):
            if not callable(getattr(processor, method, None)):
                raise TypeError(f"SAM3 processor must define callable {method}()")
        self.processor = processor
        self.processor.set_confidence_threshold(0.0)

    def set_image(self, rgb):
        return self.processor.set_image(Image.fromarray(rgb))

    def segment(self, state, object_spec):
        output = self.processor.set_text_prompt(
            state=state,
            prompt=object_spec.prompt,
        )
        for key in ("masks", "boxes", "scores"):
            if key not in output:
                raise KeyError(f"SAM3 output is missing {key!r}")
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.detach().cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        masks = np.asarray(masks)
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(f"SAM3 masks must have shape [N, H, W], got {masks.shape}")
        expected_image_shape = (
            int(state.get("original_height", masks.shape[1])),
            int(state.get("original_width", masks.shape[2])),
        )
        if masks.shape[1:] != expected_image_shape:
            raise ValueError(
                f"SAM3 masks have image shape {masks.shape[1:]}, expected "
                f"{expected_image_shape}"
            )
        if boxes.shape != (len(masks), 4):
            raise ValueError(
                f"SAM3 boxes must have shape ({len(masks)}, 4), got {boxes.shape}"
            )
        if scores.shape != (len(masks),):
            raise ValueError(
                f"SAM3 scores must have shape ({len(masks)},), got {scores.shape}"
            )
        if len(masks) == 0:
            raise RuntimeError(
                f"SAM3 found no instances for object {object_spec.name!r} "
                f"with prompt {object_spec.prompt!r}"
            )
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(boxes)):
            raise ValueError("SAM3 boxes and scores must contain only finite values")
        if object_spec.selection == "highest_score":
            selected_index = int(np.argmax(scores))
        else:
            selected_index = object_spec.instance_index
            if selected_index >= len(masks):
                raise IndexError(
                    f"Object {object_spec.name!r} requested SAM3 instance "
                    f"{selected_index}, but only {len(masks)} were found"
                )
        score = float(scores[selected_index])
        if score < object_spec.min_score:
            raise RuntimeError(
                f"SAM3 score {score:.4f} for object {object_spec.name!r} is below "
                f"required min_score {object_spec.min_score:.4f}"
            )
        mask = np.asarray(masks[selected_index], dtype=bool)
        if not mask.any():
            raise RuntimeError(
                f"SAM3 selected an empty mask for object {object_spec.name!r}"
            )
        return SegmentationResult(
            mask=mask,
            box_xyxy=boxes[selected_index],
            score=score,
            instance_index=selected_index,
        )


class RealWorldMapConstructor:
    """Construct one Map4D frame with SAM3 masks and FoundationPose poses."""

    def __init__(
        self,
        *,
        task_name,
        sam3_processor,
        foundationpose_factory: Callable[[ObjectSpec], Any],
        metadata_root=REALWORLD_DIR,
        metadata_path=None,
        output_device="cpu",
        map_builder: Optional[
            Callable[[TaskSpec, Sequence[ObjectPoseResult], torch.device], Any]
        ] = None,
    ):
        if not callable(foundationpose_factory):
            raise TypeError("foundationpose_factory must be callable")
        if map_builder is not None and not callable(map_builder):
            raise TypeError("map_builder must be callable")
        self.task_spec = load_realworld_task(
            task_name,
            metadata_root=metadata_root,
            metadata_path=metadata_path,
        )
        self.segmenter = SAM3Segmenter(sam3_processor)
        self.foundationpose_factory = foundationpose_factory
        self.output_device = torch.device(output_device)
        self.map_builder = map_builder
        self.representation_types = (
            _representation_types() if map_builder is None else None
        )
        self.estimators = {}
        for object_spec in self.task_spec.objects:
            estimator = foundationpose_factory(object_spec)
            if not callable(getattr(estimator, "register", None)):
                raise TypeError(
                    f"FoundationPose estimator for {object_spec.name!r} must define "
                    "callable register()"
                )
            if not hasattr(estimator, "pose_last"):
                raise TypeError(
                    f"FoundationPose estimator for {object_spec.name!r} must expose "
                    "pose_last so registration fallback can be detected"
                )
            self.estimators[object_spec.name] = estimator

    def construct(
        self,
        *,
        rgb,
        depth,
        camera_intrinsics,
        output_from_camera,
    ):
        rgb = _as_rgb_uint8(rgb)
        depth = _as_depth_float32(depth, rgb.shape)
        camera_intrinsics = _as_intrinsics(camera_intrinsics)
        output_from_camera = _as_transform(
            output_from_camera,
            "output_from_camera",
        )
        sam3_state = self.segmenter.set_image(rgb)
        object_results = []
        for object_spec in self.task_spec.objects:
            segmentation = self.segmenter.segment(sam3_state, object_spec)
            valid_depth = (
                segmentation.mask
                & np.isfinite(depth)
                & (depth >= 0.001)
            )
            valid_depth_count = int(valid_depth.sum())
            if valid_depth_count < FOUNDATIONPOSE_MIN_VALID_DEPTH_PIXELS:
                raise RuntimeError(
                    f"Object {object_spec.name!r} mask contains "
                    f"{valid_depth_count} valid depth pixels; FoundationPose "
                    f"requires at least {FOUNDATIONPOSE_MIN_VALID_DEPTH_PIXELS}"
                )
            estimator = self.estimators[object_spec.name]
            estimator.pose_last = None
            pose = estimator.register(
                K=camera_intrinsics,
                rgb=rgb,
                depth=depth,
                ob_mask=segmentation.mask.astype(np.uint8),
                iteration=object_spec.refine_iterations,
            )
            if estimator.pose_last is None:
                raise RuntimeError(
                    f"FoundationPose registration failed for object "
                    f"{object_spec.name!r}; estimator.pose_last was not set"
                )
            camera_from_object = _as_transform(
                pose,
                f"FoundationPose result for {object_spec.name!r}",
            )
            object_results.append(
                ObjectPoseResult(
                    object_spec=object_spec,
                    segmentation=segmentation,
                    camera_from_object=camera_from_object,
                    output_from_object=output_from_camera @ camera_from_object,
                )
            )

        if self.map_builder is None:
            map4d = self._build_generic_map(object_results)
        else:
            map4d = self.map_builder(
                self.task_spec,
                tuple(object_results),
                self.output_device,
            )
        positions = torch.as_tensor(
            np.concatenate(
                [result.output_from_object[:3, 3] for result in object_results]
            )[None],
            dtype=torch.float32,
            device=self.output_device,
        )
        rotations = torch.as_tensor(
            np.concatenate(
                [
                    _matrix_to_rotation_6d(result.output_from_object[:3, :3])
                    for result in object_results
                ]
            )[None],
            dtype=torch.float32,
            device=self.output_device,
        )
        sizes = torch.tensor(
            [self.task_spec.size_parameters],
            dtype=torch.float32,
            device=self.output_device,
        )
        return ConstructionResult(
            task_spec=self.task_spec,
            map4d=map4d,
            size_parameters=sizes,
            positions=positions,
            rotations_6d=rotations,
            objects=tuple(object_results),
        )

    def _build_generic_map(self, object_results):
        if self.representation_types is None:
            raise RuntimeError("Generic Map4D representation was not initialized")
        geometry, object_class, map_class, structure_node = self.representation_types
        objects = []
        for result in object_results:
            size_values = {
                parameter.name: self.task_spec.size_parameters[
                    parameter.global_index
                ]
                for parameter in result.object_spec.size_parameters
            }
            nodes = []
            for primitive in result.object_spec.primitives:
                primitive_class = getattr(geometry, primitive.primitive_type, None)
                if primitive_class is None or not isinstance(primitive_class, type):
                    raise KeyError(
                        f"Unknown geometry primitive type "
                        f"{primitive.primitive_type!r}"
                    )
                parameter_values = {}
                for name, source in primitive.parameters.items():
                    value = size_values[source] if isinstance(source, str) else float(source)
                    parameter_values[name] = torch.tensor(
                        [value],
                        dtype=torch.float32,
                        device=self.output_device,
                    )
                output_from_primitive = (
                    result.output_from_object
                    @ _pose_from_local(
                        primitive.local_position,
                        primitive.local_rotation_6d,
                    )
                )
                position = torch.as_tensor(
                    output_from_primitive[:3, 3][None],
                    dtype=torch.float32,
                    device=self.output_device,
                )
                rotation = torch.as_tensor(
                    _matrix_to_rotation_6d(
                        output_from_primitive[:3, :3]
                    )[None],
                    dtype=torch.float32,
                    device=self.output_device,
                )
                node = primitive_class(
                    **parameter_values,
                    **dict(primitive.arguments),
                    position=position,
                    rotation=rotation,
                    Semantic=primitive.semantic,
                    Affordance=None,
                )
                if not isinstance(node, structure_node):
                    raise TypeError(
                        f"Primitive {primitive.name!r} type "
                        f"{primitive.primitive_type!r} did not produce StructureNode"
                    )
                nodes.append(node)
            objects.append(
                object_class(
                    nodes=nodes,
                    edges=[],
                    name=result.object_spec.name,
                    semantic=result.object_spec.name,
                    prompt=result.object_spec.prompt,
                    pose_6d=result.output_from_object,
                )
            )
        return map_class(
            objects=objects,
            task_name=self.task_spec.name,
            representation_name=f"RealWorld_{self.task_spec.name}",
            metadata=self.task_spec.raw,
        )


def _representation_types():
    maps4d_dir = (
        Path(__file__).resolve().parent
        / "representation"
        / "maps4d"
    )
    maps4d_path = str(maps4d_dir)
    module_names = (
        "base_template",
        "geometry_primitive",
        "knowledge_utils",
        "utils_torch",
    )
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve().parent != maps4d_dir:
            raise RuntimeError(
                f"Cannot import Map4D representation: module {module_name!r} "
                f"is already loaded from {module_file!r}, expected {maps4d_dir}"
            )
    if maps4d_path not in sys.path:
        sys.path.insert(0, maps4d_path)
    base_template = importlib.import_module("base_template")
    geometry = importlib.import_module("geometry_primitive")
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            raise RuntimeError(
                f"Map4D representation import did not load {module_name!r}"
            )
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve().parent != maps4d_dir:
            raise RuntimeError(
                f"Map4D representation module {module_name!r} loaded from "
                f"{module_file!r}, expected {maps4d_dir}"
            )
    return (
        geometry,
        base_template.Object,
        base_template.Map_4d,
        base_template.StructureNode,
    )


__all__ = [
    "ConstructionResult",
    "ObjectPoseResult",
    "ObjectSpec",
    "PrimitiveSpec",
    "REALWORLD_DIR",
    "RealWorldMapConstructor",
    "SAM3Segmenter",
    "SegmentationResult",
    "SizeParameterSpec",
    "TaskSpec",
    "build_sam3_processor",
    "load_realworld_task",
]