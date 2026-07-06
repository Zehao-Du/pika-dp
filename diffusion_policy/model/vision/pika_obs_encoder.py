import copy
import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import resolve_data_config

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


logger = logging.getLogger(__name__)


class PikaObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str = "vit_base_patch16_clip_224.openai",
        depth_model_name: str = "resnet34.a1_in1k",
        pretrained: bool = True,
        frozen: bool = False,
        image_size: int = 224,
        depth_max: float = 2.0,
        share_rgb_model: bool = False,
        model_name=None,
        global_pool=None,
        transforms=None,
        use_group_norm=None,
        imagenet_norm=None,
        feature_aggregation=None,
        downsample_ratio=None,
        position_encording=None,
    ):
        """
        Assumes rgb input: B,T,3,H,W in [0,1].
        Assumes depth input: B,T,1,H,W in meters.
        Assumes low_dim input: B,T,D.
        """
        super().__init__()

        if depth_max <= 0:
            raise ValueError(f"depth_max must be positive, got {depth_max}.")

        rgb_keys = list()
        depth_keys = list()
        low_dim_keys = list()
        key_shape_map = dict()

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if obs_type == "rgb":
                if len(shape) != 3 or shape[0] != 3:
                    raise ValueError(f"RGB obs {key} must have shape [3,H,W], got {shape}.")
                rgb_keys.append(key)
            elif obs_type == "depth":
                if len(shape) != 3 or shape[0] != 1:
                    raise ValueError(f"Depth obs {key} must have shape [1,H,W], got {shape}.")
                depth_keys.append(key)
            elif obs_type == "low_dim":
                if not attr.get("ignore_by_policy", False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        if not rgb_keys:
            raise ValueError("PikaObsEncoder requires at least one rgb obs key.")
        if not depth_keys:
            raise ValueError("PikaObsEncoder requires at least one depth obs key.")

        rgb_model = timm.create_model(
            model_name=rgb_model_name,
            pretrained=pretrained,
            global_pool="",
            num_classes=0,
        )
        depth_model = timm.create_model(
            model_name=depth_model_name,
            pretrained=pretrained,
            in_chans=1,
            global_pool="avg",
            num_classes=0,
        )

        if frozen:
            if not pretrained:
                raise ValueError("frozen=True requires pretrained=True.")
            for param in rgb_model.parameters():
                param.requires_grad = False
            for param in depth_model.parameters():
                param.requires_grad = False

        rgb_data_config = resolve_data_config({}, model=rgb_model)
        rgb_mean = torch.tensor(rgb_data_config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        rgb_std = torch.tensor(rgb_data_config["std"], dtype=torch.float32).view(1, 3, 1, 1)

        key_rgb_model_map = nn.ModuleDict()
        for key in sorted(rgb_keys):
            key_rgb_model_map[key] = rgb_model if share_rgb_model else copy.deepcopy(rgb_model)

        key_depth_model_map = nn.ModuleDict()
        for key in sorted(depth_keys):
            key_depth_model_map[key] = copy.deepcopy(depth_model)

        self.shape_meta = shape_meta
        self.rgb_model_name = rgb_model_name
        self.depth_model_name = depth_model_name
        self.image_size = int(image_size)
        self.depth_max = float(depth_max)
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = sorted(rgb_keys)
        self.depth_keys = sorted(depth_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map
        self.key_rgb_model_map = key_rgb_model_map
        self.key_depth_model_map = key_depth_model_map
        self.register_buffer("rgb_mean", rgb_mean)
        self.register_buffer("rgb_std", rgb_std)

        print("rgb keys:          ", self.rgb_keys)
        print("depth keys:        ", self.depth_keys)
        print("low_dim_keys keys: ", self.low_dim_keys)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _resize(self, x):
        if x.shape[-2:] == (self.image_size, self.image_size):
            return x
        return F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _encode_rgb(self, key, img):
        B, T = img.shape[:2]
        img = img.reshape(B * T, *img.shape[2:])
        img = self._resize(img)
        img = (img - self.rgb_mean.to(dtype=img.dtype)) / self.rgb_std.to(dtype=img.dtype)

        raw_feature = self.key_rgb_model_map[key](img)
        if raw_feature.ndim != 3:
            raise RuntimeError(
                f"RGB model {self.rgb_model_name} must return token features with shape [B,N,D], "
                f"got {raw_feature.shape}."
            )
        feature = raw_feature[:, 0, :]
        return feature.reshape(B, -1)

    def _encode_depth(self, key, depth):
        B, T = depth.shape[:2]
        depth = depth.reshape(B * T, *depth.shape[2:])
        depth = torch.clamp(depth, min=0.0, max=self.depth_max) / self.depth_max
        depth = self._resize(depth)

        feature = self.key_depth_model_map[key](depth)
        if feature.ndim != 2:
            raise RuntimeError(
                f"Depth model {self.depth_model_name} must return pooled features with shape [B,D], "
                f"got {feature.shape}."
            )
        return feature.reshape(B, -1)

    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        for key in self.rgb_keys:
            if key not in obs_dict:
                raise KeyError(f"Missing rgb obs key: {key}")
            img = obs_dict[key]
            B = img.shape[0]
            if B != batch_size:
                raise ValueError(f"Batch size mismatch for {key}: {B} != {batch_size}")
            if tuple(img.shape[2:]) != self.key_shape_map[key]:
                raise ValueError(f"Shape mismatch for {key}: {tuple(img.shape[2:])} != {self.key_shape_map[key]}")
            features.append(self._encode_rgb(key, img))

        for key in self.depth_keys:
            if key not in obs_dict:
                raise KeyError(f"Missing depth obs key: {key}")
            depth = obs_dict[key]
            B = depth.shape[0]
            if B != batch_size:
                raise ValueError(f"Batch size mismatch for {key}: {B} != {batch_size}")
            if tuple(depth.shape[2:]) != self.key_shape_map[key]:
                raise ValueError(f"Shape mismatch for {key}: {tuple(depth.shape[2:])} != {self.key_shape_map[key]}")
            features.append(self._encode_depth(key, depth))

        for key in self.low_dim_keys:
            if key not in obs_dict:
                raise KeyError(f"Missing low_dim obs key: {key}")
            data = obs_dict[key]
            B = data.shape[0]
            if B != batch_size:
                raise ValueError(f"Batch size mismatch for {key}: {B} != {batch_size}")
            if tuple(data.shape[2:]) != self.key_shape_map[key]:
                raise ValueError(f"Shape mismatch for {key}: {tuple(data.shape[2:])} != {self.key_shape_map[key]}")
            features.append(data.reshape(B, -1))

        if not features:
            raise RuntimeError("No observation features were produced.")
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros(
                (1, attr["horizon"]) + shape,
                dtype=self.dtype,
                device=self.device,
            )
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        if example_output.ndim != 2 or example_output.shape[0] != 1:
            raise RuntimeError(f"Unexpected output shape: {example_output.shape}")
        return example_output.shape
