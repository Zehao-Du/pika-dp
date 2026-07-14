import json
import pathlib
import sys
import tempfile
import unittest

import torch


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.model.mappolicy.map_encoder import (
    MAP_NODE_VOCAB,
    Map4DEncoder,
    load_maps4d_vocab,
)


IDENTITY_ROTATION_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


class _Node:
    def __init__(self, position, semantic, point_count=4):
        batch_size = position.shape[0]
        self.position = position
        self.rotation = IDENTITY_ROTATION_6D.repeat(batch_size, 1)
        self.Node_Position = position[:, None].repeat(1, point_count, 1)
        self.Node_Affordance = None
        self.Node_Semantic = semantic


class _Map:
    def __init__(self, nodes):
        self.Nodes = nodes
        self.Edges = []


class MapEncoderTest(unittest.TestCase):
    def test_bundled_metadata_vocab(self):
        self.assertEqual(MAP_NODE_VOCAB["StackCube-v1"], 3)
        self.assertEqual(MAP_NODE_VOCAB["Map4d_StackCube"], 3)
        self.assertEqual(MAP_NODE_VOCAB["PlugCharger-v1"], 5)

    def test_custom_metadata_vocab(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "task.json"
            path.write_text(
                json.dumps(
                    {
                        "Task-v1": {
                            "representation": "TaskMap",
                            "graph": {
                                "node_count": 2,
                                "node_semantics": ["part a", "part b"],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            node_vocab, semantic_vocab = load_maps4d_vocab(tmp_dir)
        self.assertEqual(node_vocab["Task-v1"], 2)
        self.assertEqual(node_vocab["TaskMap"], 2)
        self.assertIn("part a", semantic_vocab)

    def test_text_semantic_forward_and_aux_loss(self):
        batch_size = 2
        positions = [
            torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.2, 0.0], [0.1, 0.2, 0.0]]),
            torch.tensor([[0.0, 0.0, -0.1], [0.1, 0.0, -0.1]]),
        ]
        map4d = _Map(
            [
                _Node(positions[0], "red cube"),
                _Node(positions[1], "green cube"),
                _Node(positions[2], "desk"),
            ]
        )
        encoder = Map4DEncoder(
            map_name="StackCube-v1",
            hidden_dim=12,
            num_layers=2,
            num_heads=3,
            dropout=0.0,
        )
        encoder.eval()
        output = encoder(map4d)
        self.assertEqual(output.shape, (batch_size, 3, 15))
        torch.testing.assert_close(output[:, :, :3], torch.stack(positions, dim=1))

        encoder.train()
        output, losses = encoder(map4d, return_loss=True)
        self.assertEqual(output.shape, (batch_size, 3, 15))
        self.assertEqual(set(losses), {"math_loss", "ortho_loss"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_tensor_semantic_forward_without_pyg_specific_input(self):
        batch_size = 2
        semantic_dim = 4
        nodes = [
            _Node(torch.zeros(batch_size, 3), torch.ones(batch_size, semantic_dim)),
            _Node(torch.ones(batch_size, 3), torch.zeros(batch_size, semantic_dim)),
        ]
        encoder = Map4DEncoder(
            num_nodes=2,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            sem_dim=semantic_dim,
            semantic_vocab={"<tensor>": 0},
            dropout=0.0,
        )
        output = encoder(_Map(nodes))
        self.assertEqual(output.shape, (batch_size, 2, 11))

    def test_invalid_attention_shape_fails_early(self):
        with self.assertRaisesRegex(ValueError, "divide hidden_dim"):
            Map4DEncoder(num_nodes=2, hidden_dim=10, num_heads=3)

    def test_sparse_semantic_vocab_fails_early(self):
        with self.assertRaisesRegex(ValueError, "unique and contiguous"):
            Map4DEncoder(
                num_nodes=2,
                hidden_dim=8,
                num_heads=2,
                semantic_vocab={"<tensor>": 0, "part": 2},
            )


if __name__ == "__main__":
    unittest.main()
