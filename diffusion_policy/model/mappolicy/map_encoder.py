import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_max


TYPE_VOCAB = {
    "Free": 0,
    "Fixed": 1,
    "Revolute": 2,
    "Prismatic": 3,
    "Cylindrical": 4,
    "Planar-Contact": 5,
    "Alignment": 6,
}

def _normalize_semantic_label(label):
    return " ".join(str(label).strip().lower().split())


def _maps4d_metadata_dir():
    return Path(__file__).resolve().parent / "representation" / "maps4d"


def _load_maps4d_vocab(metadata_dir=None):
    metadata_dir = Path(metadata_dir) if metadata_dir is not None else _maps4d_metadata_dir()
    map_node_vocab = {}
    semantic_labels = ["<tensor>"]

    for path in sorted(metadata_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        for task_name, task_meta in payload.items():
            graph_meta = task_meta.get("graph")
            if graph_meta is None or "node_semantics" not in graph_meta:
                raise KeyError(f"{path} task {task_name!r} must define graph.node_semantics")

            node_semantics = [_normalize_semantic_label(item) for item in graph_meta["node_semantics"]]
            if len(node_semantics) == 0:
                raise ValueError(f"{path} task {task_name!r} has empty graph.node_semantics")

            node_count = int(graph_meta.get("node_count", len(node_semantics)))
            if node_count != len(node_semantics):
                raise ValueError(
                    f"{path} task {task_name!r}: graph.node_count={node_count} "
                    f"but graph.node_semantics has {len(node_semantics)} entries"
                )

            keys = [task_name, path.stem]
            representation = task_meta.get("representation")
            if representation:
                keys.append(representation)
            for key in keys:
                map_node_vocab[key] = node_count

            semantic_labels.extend(node_semantics)

    if len(map_node_vocab) == 0:
        raise RuntimeError(f"No maps4d metadata json files found in {metadata_dir}")

    semantic_vocab = {}
    for label in semantic_labels:
        if label not in semantic_vocab:
            semantic_vocab[label] = len(semantic_vocab)
    return map_node_vocab, semantic_vocab


def load_maps4d_vocab(metadata_dir=None):
    """Load task node counts and semantic labels from Map4D metadata JSON."""
    return _load_maps4d_vocab(metadata_dir=metadata_dir)


MAP_NODE_VOCAB, SEMANTIC_NODE_VOCAB = _load_maps4d_vocab()


def _to_tensor(value, device=None, dtype=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device if device is not None else tensor.device, dtype=dtype if dtype is not None else tensor.dtype)
    return tensor


def _first_tensor_from_nodes(nodes):
    for node in nodes:
        for name in ("position", "rotation", "Node_Position", "Node_Affordance"):
            value = getattr(node, name, None)
            if torch.is_tensor(value):
                return value
    raise ValueError("Cannot infer batch/device from an empty or tensor-free Map4D graph")


def _node_batch_size(node):
    for name in ("position", "rotation", "Node_Position", "Node_Affordance"):
        value = getattr(node, name, None)
        if torch.is_tensor(value):
            return value.shape[0]
    raise ValueError("Map4D node must contain batched tensor attributes")


def _node_device_dtype(node):
    tensor = _first_tensor_from_nodes([node])
    return tensor.device, tensor.dtype


def _as_points(value, batch_size, device, dtype):
    if value is None:
        return None
    points = _to_tensor(value, device=device, dtype=dtype)
    if points.ndim == 2 and points.shape[-1] == 3:
        points = points.unsqueeze(0).expand(batch_size, -1, -1)
    if points.ndim != 3 or points.shape[0] != batch_size or points.shape[-1] != 3:
        raise ValueError(f"Expected point tensor [B, P, 3], got {tuple(points.shape)}")
    return points


def _node_pose_tensor(value, batch_size, device, dtype, name):
    tensor = _to_tensor(value, device=device, dtype=dtype)
    if tensor is None:
        raise ValueError(f"Map4D node must define {name}")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).expand(batch_size, -1)
    if tensor.ndim != 2 or tensor.shape[0] != batch_size:
        raise ValueError(f"Expected node {name} tensor [B, D], got {tuple(tensor.shape)}")
    return tensor


def _semantic_id(raw_semantic, semantic_vocab=None):
    if torch.is_tensor(raw_semantic):
        return None
    semantic_vocab = semantic_vocab or SEMANTIC_NODE_VOCAB
    key = _normalize_semantic_label(raw_semantic)
    if key not in semantic_vocab:
        raise KeyError(f"Unknown Map4D node semantic {raw_semantic!r}")
    return semantic_vocab[key]


def _semantic_embedding_tensor(raw_semantic, batch_size, device):
    if not torch.is_tensor(raw_semantic):
        return None
    sem = raw_semantic.to(device=device, dtype=torch.float32)
    if sem.ndim == 1:
        sem = sem.unsqueeze(0).expand(batch_size, -1)
    if sem.ndim != 2 or sem.shape[0] != batch_size:
        raise ValueError(f"Expected semantic tensor [B, C], got {tuple(sem.shape)}")
    return sem


def _flatten_anchor(anchor_list, batch_size, device, dtype):
    flat = []
    for anchor in anchor_list:
        if anchor is None:
            flat.append(torch.zeros(batch_size, 12, device=device, dtype=dtype))
            continue
        p = _to_tensor(anchor.get("p"), device=device, dtype=dtype)
        if p is None:
            flat.append(torch.zeros(batch_size, 12, device=device, dtype=dtype))
            continue
        zero = torch.zeros_like(p)
        primary = anchor.get("n", anchor.get("d", None))
        primary = _to_tensor(primary, device=device, dtype=dtype) if primary is not None else zero
        tangent = _to_tensor(anchor.get("t"), device=device, dtype=dtype) if anchor.get("t") is not None else zero
        bitangent = _to_tensor(anchor.get("b"), device=device, dtype=dtype) if anchor.get("b") is not None else zero
        flat.append(torch.cat([p, primary, tangent, bitangent], dim=-1))

    if len(flat) == 0:
        return torch.zeros(batch_size, 24, device=device, dtype=dtype)
    out = torch.cat(flat, dim=-1)
    if out.shape[-1] < 24:
        pad = torch.zeros(batch_size, 24 - out.shape[-1], device=device, dtype=dtype)
        out = torch.cat([out, pad], dim=-1)
    return out[..., :24]


def _edge_type_id(edge):
    raw_type = getattr(edge, "C_Type", "Free")
    if torch.is_tensor(raw_type):
        return raw_type
    if str(raw_type) not in TYPE_VOCAB:
        raise KeyError(f"Unknown Map4D edge type {raw_type!r}")
    return TYPE_VOCAB[str(raw_type)]


def _edge_param(edge, batch_size, device, dtype):
    param = getattr(edge, "Parameter", None)
    if param is None:
        return torch.zeros(batch_size, 3, device=device, dtype=dtype)
    param = _to_tensor(param, device=device, dtype=dtype)
    if param.ndim == 1:
        param = param.unsqueeze(0).expand(batch_size, -1)
    if param.shape[-1] < 3:
        pad = torch.zeros(batch_size, 3 - param.shape[-1], device=device, dtype=dtype)
        param = torch.cat([param, pad], dim=-1)
    return param[..., :3]


def _edge_pose(edge, nodes, batch_size, device, dtype):
    pose = getattr(edge, "Relative_pose", None)
    if pose is None:
        pose = getattr(edge, "Relative_Pose", None)
    if pose is not None:
        pose = _to_tensor(pose, device=device, dtype=dtype)
        if pose.ndim == 1:
            pose = pose.unsqueeze(0).expand(batch_size, -1)
        if pose.shape[-1] < 9:
            pad = torch.zeros(batch_size, 9 - pose.shape[-1], device=device, dtype=dtype)
            pose = torch.cat([pose, pad], dim=-1)
        return pose[..., :9]

    src_idx, dst_idx = getattr(edge, "Node_idx", (0, 0))
    src = nodes[src_idx]
    dst = nodes[dst_idx]
    src_pos = _to_tensor(getattr(src, "position"), device=device, dtype=dtype)
    dst_pos = _to_tensor(getattr(dst, "position"), device=device, dtype=dtype)
    src_rot = _to_tensor(getattr(src, "rotation"), device=device, dtype=dtype)
    dst_rot = _to_tensor(getattr(dst, "rotation"), device=device, dtype=dtype)
    rel_pos = dst_pos - src_pos
    rel_rot = dst_rot - src_rot
    if rel_rot.shape[-1] < 6:
        rel_rot = F.pad(rel_rot, (0, 6 - rel_rot.shape[-1]))
    return torch.cat([rel_pos, rel_rot[..., :6]], dim=-1)


def _edge_anchor(edge, nodes, batch_size, device, dtype):
    anchors = getattr(edge, "Anchor", None)
    if anchors:
        return _flatten_anchor(anchors, batch_size, device, dtype)

    ref = getattr(edge, "Refrence_Anchor", None)
    node_idx = getattr(edge, "Node_idx", None)
    if not isinstance(ref, dict) or node_idx is None:
        return torch.zeros(batch_size, 24, device=device, dtype=dtype)

    anchors = []
    for idx in node_idx:
        ref_spec = ref.get(idx)
        if ref_spec is None:
            anchors.append(None)
            continue
        node = nodes[idx]
        anchor_groups = getattr(node, "Refrence_Anchor", None)
        anchors.append(anchor_groups[ref_spec["type"]][ref_spec["idx"]])
    return _flatten_anchor(anchors, batch_size, device, dtype)


def _complete_free_edges(num_nodes):
    for src in range(num_nodes):
        for dst in range(src + 1, num_nodes):
            yield src, dst


def build_map4d_graph_data(map4d, semantic_vocab=None, complete_graph=True):
    """Build the batched graph consumed by RoboGraphormer from maps4d objects."""
    nodes = getattr(map4d, "Nodes", None)
    if nodes is None:
        nodes = getattr(map4d, "Node", None)
    if nodes is None:
        raise ValueError("Map4D representation must expose Nodes or Node")
    nodes = list(nodes)
    if len(nodes) == 0:
        raise ValueError("Map4D representation contains no nodes")

    first = nodes[0]
    batch_size = _node_batch_size(first)
    device, dtype = _node_device_dtype(first)
    num_nodes_per_graph = len(nodes)
    total_nodes = batch_size * num_nodes_per_graph
    semantic_vocab = semantic_vocab or SEMANTIC_NODE_VOCAB

    all_pos_points = []
    all_pos_indices = []
    all_aff_points = []
    all_aff_indices = []
    sem_ids = []
    sem_embeddings = []
    node_positions = []
    node_rotations = []
    saw_tensor_semantic = False
    saw_text_semantic = False

    for local_idx, node in enumerate(nodes):
        global_ids = torch.arange(batch_size, device=device) * num_nodes_per_graph + local_idx
        node_positions.append(_node_pose_tensor(getattr(node, "position", None), batch_size, device, dtype, "position"))
        node_rotations.append(_node_pose_tensor(getattr(node, "rotation", None), batch_size, device, dtype, "rotation"))

        pos_points = _as_points(getattr(node, "Node_Position", None), batch_size, device, dtype)
        if pos_points is not None:
            all_pos_points.append(pos_points.reshape(-1, 3))
            all_pos_indices.append(global_ids.unsqueeze(1).expand(batch_size, pos_points.shape[1]).reshape(-1))

        aff_points = _as_points(getattr(node, "Node_Affordance", None), batch_size, device, dtype)
        if aff_points is not None:
            all_aff_points.append(aff_points.reshape(-1, 3))
            all_aff_indices.append(global_ids.unsqueeze(1).expand(batch_size, aff_points.shape[1]).reshape(-1))

        raw_semantic = getattr(node, "Node_Semantic", None)
        sem_id = _semantic_id(raw_semantic, semantic_vocab=semantic_vocab)
        if sem_id is None:
            sem_id = semantic_vocab["<tensor>"]
        sem_ids.append(torch.full((batch_size,), sem_id, device=device, dtype=torch.long))

        sem_tensor = _semantic_embedding_tensor(raw_semantic, batch_size, device)
        if sem_tensor is None:
            saw_text_semantic = True
        else:
            saw_tensor_semantic = True
            sem_embeddings.append(sem_tensor)

    if all_pos_points:
        x_pos = torch.cat(all_pos_points, dim=0)
        pos_idx = torch.cat(all_pos_indices, dim=0).long()
    else:
        x_pos = torch.empty(0, 3, device=device, dtype=dtype)
        pos_idx = torch.empty(0, device=device, dtype=torch.long)

    if all_aff_points:
        x_aff = torch.cat(all_aff_points, dim=0)
        aff_idx = torch.cat(all_aff_indices, dim=0).long()
    else:
        x_aff = torch.empty(0, 3, device=device, dtype=dtype)
        aff_idx = torch.empty(0, device=device, dtype=torch.long)

    x_sem_id = torch.stack(sem_ids, dim=1).reshape(-1)
    node_pos = torch.stack(node_positions, dim=1).reshape(total_nodes, -1)
    node_rot = torch.stack(node_rotations, dim=1).reshape(total_nodes, -1)
    if saw_tensor_semantic and saw_text_semantic:
        raise ValueError("Map4D node semantics must be all text labels or all embedding tensors")
    x_sem = torch.stack(sem_embeddings, dim=1).reshape(total_nodes, -1) if saw_tensor_semantic else None

    explicit_edges = list(getattr(map4d, "Edges", getattr(map4d, "Edge", [])))
    edge_specs = []
    existing = set()
    for edge in explicit_edges:
        src, dst = getattr(edge, "Node_idx", (None, None))
        if src is None or dst is None:
            continue
        edge_specs.append((src, dst, edge))
        existing.add(tuple(sorted((src, dst))))

    if complete_graph:
        for src, dst in _complete_free_edges(num_nodes_per_graph):
            if (src, dst) not in existing:
                edge_specs.append((src, dst, None))

    base_src = []
    base_dst = []
    edge_type_chunks = []
    edge_param_chunks = []
    edge_anchor_chunks = []
    edge_pose_chunks = []

    for src, dst, edge in edge_specs:
        base_src.append(src)
        base_dst.append(dst)

        if edge is None:
            edge_type = torch.full((batch_size, 1), TYPE_VOCAB["Free"], device=device, dtype=torch.long)
            edge_param = torch.zeros(batch_size, 3, device=device, dtype=dtype)
            edge_anchor = torch.zeros(batch_size, 24, device=device, dtype=dtype)
            src_node = nodes[src]
            dst_node = nodes[dst]
            rel_pos = _to_tensor(getattr(dst_node, "position"), device=device, dtype=dtype) - _to_tensor(
                getattr(src_node, "position"), device=device, dtype=dtype
            )
            rel_rot = _to_tensor(getattr(dst_node, "rotation"), device=device, dtype=dtype) - _to_tensor(
                getattr(src_node, "rotation"), device=device, dtype=dtype
            )
            if rel_rot.shape[-1] < 6:
                rel_rot = F.pad(rel_rot, (0, 6 - rel_rot.shape[-1]))
            edge_pose = torch.cat([rel_pos, rel_rot[..., :6]], dim=-1)
        else:
            edge_type = _edge_type_id(edge)
            if torch.is_tensor(edge_type):
                edge_type = edge_type.to(device=device).view(batch_size, 1).long()
            else:
                edge_type = torch.full((batch_size, 1), edge_type, device=device, dtype=torch.long)
            edge_param = _edge_param(edge, batch_size, device, dtype)
            edge_anchor = _edge_anchor(edge, nodes, batch_size, device, dtype)
            edge_pose = _edge_pose(edge, nodes, batch_size, device, dtype)

        edge_type_chunks.append(edge_type)
        edge_param_chunks.append(edge_param)
        edge_anchor_chunks.append(edge_anchor)
        edge_pose_chunks.append(edge_pose)

    if edge_type_chunks:
        base_edge_index = torch.tensor([base_src, base_dst], device=device, dtype=torch.long)
        offsets = (torch.arange(batch_size, device=device) * num_nodes_per_graph).view(batch_size, 1, 1)
        batched_edges = base_edge_index.unsqueeze(0) + offsets
        edge_index = batched_edges.permute(1, 0, 2).reshape(2, -1)
        edge_type = torch.stack(edge_type_chunks, dim=1).reshape(-1, 1)
        edge_param = torch.stack(edge_param_chunks, dim=1).reshape(-1, 3)
        edge_anchor = torch.stack(edge_anchor_chunks, dim=1).reshape(-1, 24)
        edge_pose = torch.stack(edge_pose_chunks, dim=1).reshape(-1, 9)
    else:
        edge_index = torch.empty(2, 0, device=device, dtype=torch.long)
        edge_type = torch.empty(0, 1, device=device, dtype=torch.long)
        edge_param = torch.empty(0, 3, device=device, dtype=dtype)
        edge_anchor = torch.empty(0, 24, device=device, dtype=dtype)
        edge_pose = torch.empty(0, 9, device=device, dtype=dtype)

    batch = torch.arange(batch_size, device=device).repeat_interleave(num_nodes_per_graph)
    payload = dict(
        x_sem=x_sem,
        x_sem_id=x_sem_id,
        node_pos=node_pos,
        node_rot=node_rot,
        x_pos=x_pos,
        pos_batch_idx=pos_idx,
        x_aff=x_aff,
        aff_batch_idx=aff_idx,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_param=edge_param,
        edge_anchor=edge_anchor,
        edge_pose=edge_pose,
        raw_edge_type=edge_type.clone(),
        raw_edge_param=edge_param.clone(),
        num_nodes=total_nodes,
        batch=batch,
        num_graphs=batch_size,
        nodes_per_graph=torch.full((batch_size,), num_nodes_per_graph, device=device, dtype=torch.long),
    )
    return Data(**payload)


def _data_to_device(data, device):
    if hasattr(data, "to") and callable(data.to):
        return data.to(device)
    for name, value in list(vars(data).items()):
        if torch.is_tensor(value):
            setattr(data, name, value.to(device))
    return data


def _coerce_graph_data(map_input, semantic_vocab=None, complete_graph=True, device=None):
    if hasattr(map_input, "to_model_data") and callable(map_input.to_model_data):
        data = map_input.to_model_data(
            semantic_vocab=semantic_vocab or SEMANTIC_NODE_VOCAB,
            complete_graph=complete_graph,
        )
    elif hasattr(map_input, "data") and getattr(map_input, "data") is not None:
        data = map_input.data
    elif hasattr(map_input, "x_pos") and hasattr(map_input, "edge_index"):
        data = map_input
    elif hasattr(map_input, "Nodes") or hasattr(map_input, "Node"):
        data = build_map4d_graph_data(map_input, semantic_vocab=semantic_vocab, complete_graph=complete_graph)
    else:
        raise TypeError("map_input must be a Map_4d/Object/Data-like object")

    return _data_to_device(data, device) if device is not None else data


class PointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, out_dim=128):
        super().__init__()
        self.out_dim = out_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, batch_idx, num_nodes):
        if x.size(0) == 0:
            return torch.zeros((num_nodes, self.out_dim), device=x.device, dtype=x.dtype)
        x = self.mlp1(x)
        x_global, _ = scatter_max(x, batch_idx.long(), dim=0, dim_size=num_nodes)
        x_global = torch.nan_to_num(x_global, nan=0.0, posinf=0.0, neginf=0.0)
        return self.mlp2(x_global)


class RoboNodeEncoder(nn.Module):
    def __init__(self, sem_dim=512, hidden_dim=768, semantic_vocab_size=None, pose_dim=9):
        super().__init__()
        self.pos_encoder = PointNetFeatureExtractor(input_dim=3, hidden_dim=64, out_dim=hidden_dim)
        self.aff_encoder = PointNetFeatureExtractor(input_dim=3, hidden_dim=64, out_dim=hidden_dim)
        self.sem_proj = nn.Linear(sem_dim, hidden_dim)
        self.sem_embed = nn.Embedding(semantic_vocab_size or len(SEMANTIC_NODE_VOCAB), hidden_dim)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, data):
        num_nodes = int(data.num_nodes)
        h_pos = self.pos_encoder(data.x_pos, data.pos_batch_idx, num_nodes)
        h_aff = self.aff_encoder(data.x_aff, data.aff_batch_idx, num_nodes)

        x_sem = getattr(data, "x_sem", None)
        if x_sem is not None:
            h_sem = self.sem_proj(x_sem.float())
        elif hasattr(data, "x_sem_id"):
            h_sem = self.sem_embed(data.x_sem_id.long())
        else:
            h_sem = torch.zeros_like(h_pos)

        if not hasattr(data, "node_pos") or not hasattr(data, "node_rot"):
            raise ValueError("Map encoder data must include node_pos and node_rot")
        h_pose = self.pose_proj(torch.cat([data.node_pos.float(), data.node_rot.float()], dim=-1))

        return self.fusion(torch.cat([h_pos, h_aff, h_sem, h_pose], dim=-1))


class RoboEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_edge_types=10):
        super().__init__()
        self.type_embed = nn.Embedding(num_edge_types, hidden_dim)
        self.param_proj = nn.Linear(3, hidden_dim)
        self.anchor_proj = nn.Linear(24, hidden_dim)
        self.pose_proj = nn.Linear(9, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, data):
        edge_type = data.edge_type.squeeze(-1).long().clamp(min=0, max=self.type_embed.num_embeddings - 1)
        h_type = self.type_embed(edge_type)
        h_param = self.param_proj(data.edge_param.float())
        h_anchor = self.anchor_proj(data.edge_anchor.float())
        h_pose = self.pose_proj(data.edge_pose.float())
        return self.fusion(torch.cat([h_type, h_param, h_anchor, h_pose], dim=-1))


class RoboGraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.edge_update_e = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update_src = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update_dst = nn.Linear(hidden_dim, hidden_dim)
        self.bias_proj = nn.Linear(hidden_dim, num_heads)
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h, e):
        batch_size, num_nodes, _ = h.shape
        h_src = self.edge_update_src(h).unsqueeze(2)
        h_dst = self.edge_update_dst(h).unsqueeze(1)
        e = F.relu(self.edge_update_e(e) + h_src + h_dst)

        attn_bias = self.bias_proj(e).permute(0, 3, 1, 2).reshape(batch_size * self.num_heads, num_nodes, num_nodes)
        residual = h
        h = self.attn_ln(h)
        h, _ = self.self_attn(h, h, h, attn_mask=attn_bias)
        h = residual + h

        residual = h
        h = self.ffn_ln(h)
        h = self.ffn(h)
        return residual + h, e


class MathConstraintHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.anchor_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 24),
        )

    def forward(self, e_final):
        return self.anchor_generator(e_final)


class RoboGraphormer(nn.Module):
    """Graph encoder for maps4d representations.

    Inputs can be:
      - a maps4d Map_4d instance;
      - a Map_4d/Object-like graph object with a populated `.data`;
      - a PyG Data already matching build_map4d_graph_data().

    The output is per-node coordinate plus feature with shape
    [B, N_node, 3 + hidden_dim].
    """

    def __init__(
        self,
        map_name=None,
        num_nodes=None,
        hidden_dim=768,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        sem_dim=512,
        include_scene_token=True,
        complete_graph=True,
        metadata_dir=None,
        map_node_vocab=None,
        semantic_vocab=None,
    ):
        super().__init__()
        if metadata_dir is not None:
            loaded_node_vocab, loaded_semantic_vocab = load_maps4d_vocab(metadata_dir)
        else:
            loaded_node_vocab, loaded_semantic_vocab = MAP_NODE_VOCAB, SEMANTIC_NODE_VOCAB
        self.map_node_vocab = dict(
            loaded_node_vocab if map_node_vocab is None else map_node_vocab
        )
        self.semantic_vocab = dict(
            loaded_semantic_vocab if semantic_vocab is None else semantic_vocab
        )
        if "<tensor>" not in self.semantic_vocab:
            raise ValueError("semantic_vocab must contain the '<tensor>' entry")
        semantic_ids = sorted(self.semantic_vocab.values())
        if semantic_ids != list(range(len(self.semantic_vocab))):
            raise ValueError(
                "semantic_vocab IDs must be unique and contiguous from 0 to "
                f"{len(self.semantic_vocab) - 1}"
            )
        if num_nodes is None and map_name is not None:
            if map_name not in self.map_node_vocab:
                raise ValueError(
                    f"Unknown map_name: {map_name}. "
                    f"Available: {list(self.map_node_vocab.keys())}"
                )
            num_nodes = self.map_node_vocab[map_name]
        if num_nodes is not None and num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide hidden_dim")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.map_name = map_name
        self.N = num_nodes
        self.hidden_dim = hidden_dim
        self.node_feature_dim = hidden_dim
        self.coordinate_dim = 3
        self.feature_dim = self.coordinate_dim + hidden_dim
        self.include_scene_token = include_scene_token
        self.complete_graph = complete_graph

        self.node_encoder = RoboNodeEncoder(
            sem_dim=sem_dim,
            hidden_dim=hidden_dim,
            semantic_vocab_size=len(self.semantic_vocab),
        )
        self.edge_encoder = RoboEdgeEncoder(hidden_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [RoboGraphormerLayer(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.math_head = MathConstraintHead(hidden_dim)

    def _batch_shape(self, data):
        if hasattr(data, "nodes_per_graph"):
            nodes_per_graph = data.nodes_per_graph
            if torch.is_tensor(nodes_per_graph):
                nodes_per_graph = nodes_per_graph.detach().cpu()
            nodes_per_graph = [int(v) for v in nodes_per_graph]
            if len(set(nodes_per_graph)) != 1:
                raise ValueError("RoboGraphormer currently requires equal nodes_per_graph within a batch")
            batch_size = len(nodes_per_graph)
            num_nodes = nodes_per_graph[0]
        elif hasattr(data, "batch"):
            batch_vec = data.batch.long()
            batch_size = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 1
            counts = torch.bincount(batch_vec, minlength=batch_size)
            if int(counts.min().item()) != int(counts.max().item()):
                raise ValueError("RoboGraphormer currently requires equal node count per graph")
            num_nodes = int(counts[0].item())
        elif self.N is not None:
            num_nodes = self.N
            batch_size = int(data.num_nodes) // num_nodes
        else:
            raise ValueError("Cannot infer graph batch shape; pass map_name/num_nodes or data.batch")

        if self.N is not None and num_nodes != self.N:
            raise ValueError(f"{self.map_name or 'map'} expects {self.N} nodes, got {num_nodes}")
        return batch_size, num_nodes

    def to_dense(self, h_sparse, e_sparse, data):
        batch_size, num_nodes = self._batch_shape(data)
        h_dense = h_sparse.view(batch_size, num_nodes, self.hidden_dim)
        e_dense = torch.zeros(
            batch_size,
            num_nodes,
            num_nodes,
            self.hidden_dim,
            device=h_sparse.device,
            dtype=h_sparse.dtype,
        )

        if data.edge_index.size(1) > 0:
            src, dst = data.edge_index.long()
            batch_idx = src // num_nodes
            local_src = src % num_nodes
            local_dst = dst % num_nodes
            e_dense[batch_idx, local_src, local_dst] = e_sparse
            e_dense[batch_idx, local_dst, local_src] = e_sparse

        return h_dense, e_dense

    def node_coordinates(self, data):
        batch_size, num_nodes = self._batch_shape(data)
        if not hasattr(data, "node_pos"):
            raise ValueError("Map encoder data must include node_pos")
        node_pos = data.node_pos.float().view(batch_size, num_nodes, -1)
        if node_pos.shape[-1] != self.coordinate_dim:
            raise ValueError(f"Expected node_pos last dim {self.coordinate_dim}, got {node_pos.shape[-1]}")
        return node_pos

    def forward(self, map_input, return_loss=False):
        device = next(self.parameters()).device
        data = _coerce_graph_data(
            map_input,
            semantic_vocab=self.semantic_vocab,
            complete_graph=self.complete_graph,
            device=device,
        )

        h_sparse = self.node_encoder(data)
        if data.edge_index.size(1) > 0:
            e_sparse = self.edge_encoder(data)
        else:
            e_sparse = torch.zeros(0, self.hidden_dim, device=h_sparse.device, dtype=h_sparse.dtype)

        h, e = self.to_dense(h_sparse, e_sparse, data)
        for layer in self.layers:
            h, e = layer(h, e)

        loss_dict = self._aux_losses(e, data) if self.training else {}
        node_output = torch.cat([self.node_coordinates(data), h], dim=-1)

        if return_loss:
            return node_output, loss_dict
        return node_output

    def encode_map(self, map_input, return_loss=False):
        return self.forward(map_input, return_loss=return_loss)

    def _aux_losses(self, e, data):
        loss_dict = {}
        if data.edge_index.size(1) == 0:
            zero = e.sum() * 0.0
            loss_dict["math_loss"] = zero
            loss_dict["ortho_loss"] = zero
            return loss_dict

        batch_size, num_nodes = self._batch_shape(data)
        dense_preds = self.math_head(e)
        src, dst = data.edge_index.long()
        batch_idx = src // num_nodes
        local_src = src % num_nodes
        local_dst = dst % num_nodes
        valid_preds = dense_preds[batch_idx, local_src, local_dst]
        math_loss, ortho_loss = self.compute_math_loss(valid_preds, data)
        loss_dict["math_loss"] = math_loss
        loss_dict["ortho_loss"] = ortho_loss
        return loss_dict

    def compute_math_loss(self, preds, data):
        n_i, t_i, b_i, p_i = preds[:, 0:3], preds[:, 3:6], preds[:, 6:9], preds[:, 9:12]
        n_j, t_j, b_j, p_j = preds[:, 12:15], preds[:, 15:18], preds[:, 18:21], preds[:, 21:24]

        n_i, t_i, b_i = F.normalize(n_i, dim=-1), F.normalize(t_i, dim=-1), F.normalize(b_i, dim=-1)
        n_j, t_j, b_j = F.normalize(n_j, dim=-1), F.normalize(t_j, dim=-1), F.normalize(b_j, dim=-1)

        def frame_loss(n, t, b):
            l_dot = (n * t).sum(-1).square() + (t * b).sum(-1).square() + (b * n).sum(-1).square()
            l_cross = torch.sum((torch.cross(n, t, dim=-1) - b).square(), dim=-1)
            return (l_dot + l_cross).mean()

        ortho_loss = frame_loss(n_i, t_i, b_i) + frame_loss(n_j, t_j, b_j)
        constraint_loss = preds.sum() * 0.0
        edge_types = data.raw_edge_type.squeeze(-1).long()
        params = data.raw_edge_param.float()

        def dot_loss(v1, v2, target):
            pred = (v1 * v2).sum(dim=-1)
            if target.dim() == 0:
                target = target.expand_as(pred)
            return F.mse_loss(pred, target)

        mask = edge_types == TYPE_VOCAB["Fixed"]
        if mask.any():
            phi_cos = params[mask, 1]
            phi_sin = params[mask, 2]
            diff = p_j[mask] - p_i[mask]
            constraint_loss = constraint_loss + dot_loss(n_i[mask], n_j[mask], torch.tensor(-1.0, device=preds.device))
            constraint_loss = constraint_loss + dot_loss(t_i[mask], t_j[mask], phi_cos)
            constraint_loss = constraint_loss + dot_loss(t_i[mask], b_j[mask], -phi_sin)
            constraint_loss = constraint_loss + dot_loss(diff, n_i[mask], params[mask, 0])
            constraint_loss = constraint_loss + dot_loss(diff, t_i[mask], torch.tensor(0.0, device=preds.device))

        mask = edge_types == TYPE_VOCAB["Revolute"]
        if mask.any():
            d_i, d_j = n_i[mask], n_j[mask]
            diff = p_j[mask] - p_i[mask]
            constraint_loss = constraint_loss + torch.mean(torch.norm(torch.cross(d_i, d_j, dim=-1), dim=-1))
            constraint_loss = constraint_loss + torch.mean(torch.norm(torch.cross(diff, d_i, dim=-1), dim=-1))

        mask = edge_types == TYPE_VOCAB["Prismatic"]
        if mask.any():
            diff = p_j[mask] - p_i[mask]
            constraint_loss = constraint_loss + dot_loss(n_i[mask], n_j[mask], torch.tensor(-1.0, device=preds.device))
            constraint_loss = constraint_loss + dot_loss(t_i[mask], t_j[mask], torch.tensor(1.0, device=preds.device))
            constraint_loss = constraint_loss + dot_loss(diff, n_i[mask], params[mask, 0])
            constraint_loss = constraint_loss + dot_loss(diff, b_i[mask], torch.tensor(0.0, device=preds.device))

        mask = edge_types == TYPE_VOCAB["Cylindrical"]
        if mask.any():
            d_i, d_j = n_i[mask], n_j[mask]
            diff = p_j[mask] - p_i[mask]
            constraint_loss = constraint_loss + torch.mean(torch.norm(torch.cross(d_i, d_j, dim=-1), dim=-1))
            constraint_loss = constraint_loss + torch.mean(torch.norm(torch.cross(diff, d_i, dim=-1), dim=-1))

        mask = edge_types == TYPE_VOCAB["Planar-Contact"]
        if mask.any():
            diff = p_j[mask] - p_i[mask]
            constraint_loss = constraint_loss + dot_loss(n_i[mask], n_j[mask], torch.tensor(-1.0, device=preds.device))
            constraint_loss = constraint_loss + dot_loss(diff, n_i[mask], params[mask, 0])

        mask = edge_types == TYPE_VOCAB["Alignment"]
        if mask.any():
            cross_prod = torch.cross(n_i[mask], n_j[mask], dim=-1)
            constraint_loss = constraint_loss + torch.mean(torch.sum(cross_prod.square(), dim=-1))

        return constraint_loss, ortho_loss


Map4DEncoder = RoboGraphormer


__all__ = [
    "MAP_NODE_VOCAB",
    "SEMANTIC_NODE_VOCAB",
    "Map4DEncoder",
    "RoboGraphormer",
    "build_map4d_graph_data",
    "load_maps4d_vocab",
]
