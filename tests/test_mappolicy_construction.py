import importlib.util
import json
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np
import torch


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONSTRUCTION_PATH = (
    ROOT_DIR
    / "diffusion_policy"
    / "model"
    / "mappolicy"
    / "construction.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mappolicy_construction_under_test",
    CONSTRUCTION_PATH,
)
construction = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = construction
SPEC.loader.exec_module(construction)


def _task_payload(mesh_name):
    return {
        "PickObject": {
            "size_parameters": {
                "dim": 3,
                "default": [0.1, 0.2, 0.3],
            },
            "objects": {
                "target": {
                    "sam3": {
                        "prompt": "red target",
                        "selection": "highest_score",
                        "min_score": 0.7,
                    },
                    "foundationpose": {
                        "mesh_path": mesh_name,
                        "refine_iterations": 4,
                    },
                    "size_parameters": [
                        {
                            "name": "height",
                            "primitive": "body",
                            "global_index": 0,
                            "default": 0.1,
                        },
                        {
                            "name": "length",
                            "primitive": "body",
                            "global_index": 1,
                            "default": 0.2,
                        },
                        {
                            "name": "width",
                            "primitive": "body",
                            "global_index": 2,
                            "default": 0.3,
                        },
                    ],
                    "primitives": [
                        {
                            "name": "body",
                            "type": "Cuboid",
                            "semantic": "red target body",
                            "parameters": {
                                "height": "height",
                                "top_length": "length",
                                "top_width": "width",
                            },
                            "arguments": {},
                            "local_pose": {
                                "position": [0, 0, 0],
                                "rotation_6d": [1, 0, 0, 0, 1, 0],
                            },
                        }
                    ],
                }
            },
        }
    }


class _FakeSAM3Processor:
    def __init__(self):
        self.confidence_threshold = None

    def set_confidence_threshold(self, threshold, state=None):
        self.confidence_threshold = threshold
        return state

    def set_image(self, image):
        return {
            "original_height": image.height,
            "original_width": image.width,
        }

    def set_text_prompt(self, state, prompt):
        height = state["original_height"]
        width = state["original_width"]
        masks = torch.zeros(2, 1, height, width, dtype=torch.bool)
        masks[0, 0, 0, 0] = True
        masks[1, 0, :, :] = True
        return {
            "masks": masks,
            "boxes": torch.tensor([[0, 0, 1, 1], [0, 0, width, height]]),
            "scores": torch.tensor([0.6, 0.9]),
        }


class _FakeSAM3VideoPredictor:
    def __init__(self):
        self.requests = []
        self.frames = None

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            self.frames = request["resource_path"]
            return {"session_id": "session-1"}
        if request["type"] == "add_prompt":
            height = self.frames[0].height
            width = self.frames[0].width
            return {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_probs": np.array([0.9], dtype=np.float32),
                    "out_binary_masks": np.ones(
                        (1, height, width), dtype=bool
                    ),
                },
            }
        if request["type"] == "close_session":
            return {"is_success": True}
        raise AssertionError(f"Unexpected request: {request}")

    def handle_stream_request(self, request):
        self.requests.append(request)
        height = self.frames[0].height
        width = self.frames[0].width
        for frame_index in range(1, len(self.frames)):
            yield {
                "frame_index": frame_index,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_probs": np.array([0.9], dtype=np.float32),
                    "out_binary_masks": np.ones(
                        (1, height, width), dtype=bool
                    ),
                },
            }


class _FakeEstimator:
    def __init__(self):
        self.calls = []
        self.pose_last = None

    def register(self, **kwargs):
        self.calls.append(kwargs)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        self.pose_last = pose.copy()
        return pose


class RealWorldConstructionTest(unittest.TestCase):
    def _write_task(self, directory, payload):
        path = pathlib.Path(directory) / "PickObject.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_and_construct_object_pose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            metadata_path = self._write_task(
                tmp_dir,
                _task_payload(mesh_path.name),
            )
            estimator = _FakeEstimator()
            map_builder_calls = []

            def map_builder(task_spec, object_results, device):
                map_builder_calls.append((task_spec, object_results, device))
                return {"objects": object_results}

            processor = _FakeSAM3Processor()
            constructor = construction.RealWorldMapConstructor(
                task_name="PickObject",
                metadata_path=metadata_path,
                sam3_processor=processor,
                sam3_video_predictor=_FakeSAM3VideoPredictor(),
                foundationpose_factory=lambda object_spec: estimator,
                output_device="cpu",
                map_builder=map_builder,
            )
            output_from_camera = np.eye(4)
            output_from_camera[0, 3] = 10.0
            result = constructor.construct_from_rgbd(
                rgb=np.zeros((4, 5, 3), dtype=np.uint8),
                depth=np.ones((4, 5), dtype=np.float32),
                camera_intrinsics=np.array(
                    [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
                    dtype=np.float32,
                ),
                output_from_camera=output_from_camera,
            )

        self.assertEqual(result.task_spec.name, "PickObject")
        self.assertEqual(result.objects[0].segmentation.instance_index, 1)
        self.assertAlmostEqual(result.objects[0].segmentation.score, 0.9, places=5)
        np.testing.assert_allclose(
            result.objects[0].output_from_object[:3, 3],
            [11.0, 2.0, 3.0],
        )
        torch.testing.assert_close(
            result.size_parameters,
            torch.tensor([[0.1, 0.2, 0.3]]),
        )
        torch.testing.assert_close(
            result.positions,
            torch.tensor([[11.0, 2.0, 3.0]]),
        )
        self.assertEqual(estimator.calls[0]["iteration"], 4)
        self.assertEqual(int(estimator.calls[0]["ob_mask"].sum()), 20)
        self.assertEqual(len(map_builder_calls), 1)
        self.assertEqual(processor.confidence_threshold, 0.0)

    def test_simulator_gt_path_calls_task_map_factory_directly(self):
        calls = []

        def map_factory(**kwargs):
            calls.append(kwargs)
            return "simulator-map"

        positions = torch.tensor([[1.0, 2.0, 3.0]])
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        sizes = torch.tensor([[0.1, 0.2, 0.3]])
        relations = torch.tensor([[0.01]])

        result = construction.construct_map_from_simulator_gt(
            map_factory=map_factory,
            positions=positions,
            rotations_6d=rotations,
            size_parameters=sizes,
            relation_parameters=relations,
            preprocess=False,
        )

        self.assertEqual(result, "simulator-map")
        self.assertIs(calls[0]["positions"], positions)
        self.assertIs(calls[0]["rotations"], rotations)
        self.assertIs(calls[0]["size_parameters"], sizes)
        self.assertIs(calls[0]["relation_parameters"], relations)
        self.assertFalse(calls[0]["preprocess"])

    def test_explicit_rgbd_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            metadata_path = self._write_task(
                tmp_dir,
                _task_payload(mesh_path.name),
            )
            constructor = construction.RealWorldMapConstructor(
                task_name="PickObject",
                metadata_path=metadata_path,
                sam3_processor=_FakeSAM3Processor(),
                sam3_video_predictor=_FakeSAM3VideoPredictor(),
                foundationpose_factory=lambda object_spec: _FakeEstimator(),
                map_builder=lambda task_spec, object_results, device: "rgbd-map",
            )
            result = constructor.construct_from_rgbd(
                rgb=np.zeros((4, 5, 3), dtype=np.uint8),
                depth=np.ones((4, 5), dtype=np.float32),
                camera_intrinsics=np.array(
                    [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
                    dtype=np.float32,
                ),
                output_from_camera=np.eye(4),
            )

        self.assertEqual(result.map4d, "rgbd-map")

    def test_later_frames_use_video_tracking_without_text_prompt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            metadata_path = self._write_task(
                tmp_dir,
                _task_payload(mesh_path.name),
            )
            video_predictor = _FakeSAM3VideoPredictor()
            constructor = construction.RealWorldMapConstructor(
                task_name="PickObject",
                metadata_path=metadata_path,
                sam3_processor=_FakeSAM3Processor(),
                sam3_video_predictor=video_predictor,
                foundationpose_factory=lambda object_spec: _FakeEstimator(),
                map_builder=lambda task_spec, object_results, device: "rgbd-map",
            )
            first_frame = np.zeros((4, 5, 3), dtype=np.uint8)
            first_result = constructor.construct_from_rgbd(
                rgb=first_frame,
                depth=np.ones((4, 5), dtype=np.float32),
                camera_intrinsics=np.array(
                    [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
                    dtype=np.float32,
                ),
                output_from_camera=np.eye(4),
            )
            tracked = constructor.track_masks_after_first_frame(
                rgb_video=np.stack([first_frame, first_frame, first_frame]),
                first_frame_result=first_result,
            )

        self.assertEqual(len(tracked), 2)
        self.assertEqual(len(tracked[0]), 1)
        self.assertEqual(tracked[0][0].instance_index, 7)
        self.assertEqual(int(tracked[0][0].mask.sum()), 20)
        prompt_request = next(
            request
            for request in video_predictor.requests
            if request["type"] == "add_prompt"
        )
        self.assertNotIn("text", prompt_request)
        self.assertEqual(prompt_request["frame_index"], 0)
        self.assertIn("bounding_boxes", prompt_request)

    def test_tracking_omission_raises_without_image_fallback(self):
        class MissingFrameVideoPredictor(_FakeSAM3VideoPredictor):
            def handle_stream_request(self, request):
                self.requests.append(request)
                return iter(())

        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            metadata_path = self._write_task(
                tmp_dir,
                _task_payload(mesh_path.name),
            )
            constructor = construction.RealWorldMapConstructor(
                task_name="PickObject",
                metadata_path=metadata_path,
                sam3_processor=_FakeSAM3Processor(),
                sam3_video_predictor=MissingFrameVideoPredictor(),
                foundationpose_factory=lambda object_spec: _FakeEstimator(),
                map_builder=lambda task_spec, object_results, device: "rgbd-map",
            )
            first_frame = np.zeros((4, 5, 3), dtype=np.uint8)
            first_result = constructor.construct_from_rgbd(
                rgb=first_frame,
                depth=np.ones((4, 5), dtype=np.float32),
                camera_intrinsics=np.array(
                    [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
                    dtype=np.float32,
                ),
                output_from_camera=np.eye(4),
            )
            with self.assertRaisesRegex(RuntimeError, "omitted frames"):
                constructor.track_masks_after_first_frame(
                    rgb_video=[first_frame, first_frame],
                    first_frame_result=first_result,
                )

    def test_schema_requires_explicit_selection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            payload = _task_payload(mesh_path.name)
            del payload["PickObject"]["objects"]["target"]["sam3"]["selection"]
            metadata_path = self._write_task(tmp_dir, payload)
            with self.assertRaisesRegex(ValueError, "selection"):
                construction.load_realworld_task(
                    "PickObject",
                    metadata_path=metadata_path,
                )

    def test_mask_without_valid_depth_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            metadata_path = self._write_task(
                tmp_dir,
                _task_payload(mesh_path.name),
            )
            constructor = construction.RealWorldMapConstructor(
                task_name="PickObject",
                metadata_path=metadata_path,
                sam3_processor=_FakeSAM3Processor(),
                sam3_video_predictor=_FakeSAM3VideoPredictor(),
                foundationpose_factory=lambda object_spec: _FakeEstimator(),
                map_builder=lambda task_spec, object_results, device: object(),
            )
            with self.assertRaisesRegex(RuntimeError, "requires at least"):
                constructor.construct_from_rgbd(
                    rgb=np.zeros((4, 5, 3), dtype=np.uint8),
                    depth=np.zeros((4, 5), dtype=np.float32),
                    camera_intrinsics=np.array(
                        [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
                        dtype=np.float32,
                    ),
                    output_from_camera=np.eye(4),
                )

    def test_rotation_6d_matches_representation_row_convention(self):
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        rotation_6d = construction._matrix_to_rotation_6d(rotation)
        np.testing.assert_allclose(
            rotation_6d,
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            construction._rotation_6d_to_matrix(rotation_6d),
            rotation,
        )

    def test_primitive_must_consume_its_assigned_parameters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            payload = _task_payload(mesh_path.name)
            parameters = payload["PickObject"]["objects"]["target"]["primitives"][0][
                "parameters"
            ]
            parameters["top_width"] = "height"
            metadata_path = self._write_task(tmp_dir, payload)
            with self.assertRaisesRegex(ValueError, "does not consume"):
                construction.load_realworld_task(
                    "PickObject",
                    metadata_path=metadata_path,
                )

    def test_boolean_is_not_accepted_as_integer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            payload = _task_payload(mesh_path.name)
            payload["PickObject"]["objects"]["target"]["foundationpose"][
                "refine_iterations"
            ] = True
            metadata_path = self._write_task(tmp_dir, payload)
            with self.assertRaisesRegex(ValueError, "JSON integer"):
                construction.load_realworld_task(
                    "PickObject",
                    metadata_path=metadata_path,
                )

    def test_duplicate_primitive_keywords_fail_during_schema_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            payload = _task_payload(mesh_path.name)
            primitive = payload["PickObject"]["objects"]["target"]["primitives"][0]
            primitive["arguments"]["height"] = 0.1
            metadata_path = self._write_task(tmp_dir, payload)
            with self.assertRaisesRegex(ValueError, "duplicate keys"):
                construction.load_realworld_task(
                    "PickObject",
                    metadata_path=metadata_path,
                )

    def test_conflicting_representation_module_fails_immediately(self):
        previous = sys.modules.get("base_template")
        conflicting = types.ModuleType("base_template")
        conflicting.__file__ = "/tmp/legacy/maps3d/base_template.py"
        sys.modules["base_template"] = conflicting
        try:
            with self.assertRaisesRegex(RuntimeError, "already loaded"):
                construction._representation_types()
        finally:
            if previous is None:
                del sys.modules["base_template"]
            else:
                sys.modules["base_template"] = previous

    def test_normalized_duplicate_object_names_fail(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            payload = _task_payload(mesh_path.name)
            target = payload["PickObject"]["objects"]["target"]
            duplicate = json.loads(json.dumps(target))
            payload["PickObject"]["objects"][" target "] = duplicate
            metadata_path = self._write_task(tmp_dir, payload)
            with self.assertRaisesRegex(ValueError, "duplicate names"):
                construction.load_realworld_task(
                    "PickObject",
                    metadata_path=metadata_path,
                )

    def test_foundationpose_fallback_pose_is_rejected(self):
        class FallbackEstimator(_FakeEstimator):
            def register(self, **kwargs):
                self.calls.append(kwargs)
                return np.eye(4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            metadata_path = self._write_task(
                tmp_dir,
                _task_payload(mesh_path.name),
            )
            constructor = construction.RealWorldMapConstructor(
                task_name="PickObject",
                metadata_path=metadata_path,
                sam3_processor=_FakeSAM3Processor(),
                sam3_video_predictor=_FakeSAM3VideoPredictor(),
                foundationpose_factory=lambda object_spec: FallbackEstimator(),
                map_builder=lambda task_spec, object_results, device: object(),
            )
            with self.assertRaisesRegex(RuntimeError, "pose_last was not set"):
                constructor.construct_from_rgbd(
                    rgb=np.zeros((4, 5, 3), dtype=np.uint8),
                    depth=np.ones((4, 5), dtype=np.float32),
                    camera_intrinsics=np.array(
                        [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
                        dtype=np.float32,
                    ),
                    output_from_camera=np.eye(4),
                )

    def test_stale_foundationpose_state_cannot_mask_later_failure(self):
        class SecondCallFallbackEstimator(_FakeEstimator):
            def register(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    pose = np.eye(4)
                    self.pose_last = pose.copy()
                    return pose
                return np.eye(4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mesh_path = pathlib.Path(tmp_dir) / "target.obj"
            mesh_path.write_text("o target\n", encoding="utf-8")
            metadata_path = self._write_task(
                tmp_dir,
                _task_payload(mesh_path.name),
            )
            estimator = SecondCallFallbackEstimator()
            constructor = construction.RealWorldMapConstructor(
                task_name="PickObject",
                metadata_path=metadata_path,
                sam3_processor=_FakeSAM3Processor(),
                sam3_video_predictor=_FakeSAM3VideoPredictor(),
                foundationpose_factory=lambda object_spec: estimator,
                map_builder=lambda task_spec, object_results, device: object(),
            )
            kwargs = {
                "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                "depth": np.ones((4, 5), dtype=np.float32),
                "camera_intrinsics": np.array(
                    [[100, 0, 2], [0, 100, 2], [0, 0, 1]],
                    dtype=np.float32,
                ),
                "output_from_camera": np.eye(4),
            }
            constructor.construct_from_rgbd(**kwargs)
            with self.assertRaisesRegex(RuntimeError, "pose_last was not set"):
                constructor.construct_from_rgbd(**kwargs)


if __name__ == "__main__":
    unittest.main()
