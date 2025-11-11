from math import sqrt
from pathlib import Path

from timm import create_model
from torch import Tensor, cat, matmul, softmax
from torch import device as torch_device
from torch import load as torch_load
from torch.cuda import is_available as torch_cuda_is_available
from torch.nn import GELU, AvgPool2d, Conv2d, LayerNorm, Linear, Module, Sequential
from torch.nn.functional import pad, unfold


class FocalSelfAttention(Module):
    def __init__(self, dim: int, num_heads: int = 8, win_size: int = 4) -> None:
        assert dim % num_heads == 0, "'dim' must be divisible by 'num_heads'"

        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = win_size
        self.qkv_proj = Linear(dim, dim * 3)
        self.out_proj = Linear(dim, dim)

        self.mid_pool = AvgPool2d(kernel_size=2, stride=2)
        self.global_pool = AvgPool2d(kernel_size=4, stride=4)

        self.token_norm = LayerNorm(dim)

    def forward(self, features: Tensor) -> Tensor:
        batch_size, channels, height, width = features.shape
        assert channels == self.dim, "channel dim mismatch"

        window_size = self.window_size
        original_features = features
        original_height, original_width = height, width

        height_padding = (window_size - height % window_size) % window_size
        width_padding = (window_size - width % window_size) % window_size

        if height_padding > 0 or width_padding > 0:
            features = pad(features, (0, width_padding, 0, height_padding))
            height, width = features.shape[2], features.shape[3]

        windows_per_col = height // window_size
        windows_per_row = width // window_size
        num_windows_total = windows_per_col * windows_per_row

        query_unfold = unfold(features, kernel_size=window_size, stride=window_size)
        query_tokens_per_window = window_size * window_size
        query_windows_tokens = query_unfold.transpose(1, 2).reshape(
            batch_size * num_windows_total, query_tokens_per_window, channels
        )
        query_windows_tokens = self.token_norm(query_windows_tokens)

        local_kernel = 2 * window_size
        padding = window_size // 2
        features_padded = pad(features, pad=(padding, padding, padding, padding))
        local_unfold = unfold(
            features_padded, kernel_size=local_kernel, stride=window_size
        )
        local_tokens_per_window = local_kernel * local_kernel
        local_kv_tokens = local_unfold.transpose(1, 2).reshape(
            batch_size * num_windows_total, local_tokens_per_window, channels
        )

        mid_pooled = self.mid_pool(features)
        mid_tokens_len = (height // 2) * (width // 2)
        mid_tokens = mid_pooled.flatten(2).transpose(1, 2)
        mid_kv_tokens_batched = (
            mid_tokens.unsqueeze(1)
            .expand(batch_size, num_windows_total, mid_tokens_len, channels)
            .reshape(batch_size * num_windows_total, mid_tokens_len, channels)
        )

        global_pooled = self.global_pool(features)
        global_tokens_len = (height // 4) * (width // 4)
        global_tokens = global_pooled.flatten(2).transpose(1, 2)
        global_kv_tokens_batched = (
            global_tokens.unsqueeze(1)
            .expand(batch_size, num_windows_total, global_tokens_len, channels)
            .reshape(batch_size * num_windows_total, global_tokens_len, channels)
        )

        concatenated_kv_tokens = cat(
            [local_kv_tokens, mid_kv_tokens_batched, global_kv_tokens_batched], dim=1
        )
        kv_tokens_per_window = concatenated_kv_tokens.shape[1]

        qkv_out_queries = self.qkv_proj(query_windows_tokens)
        queries, _, _ = qkv_out_queries.chunk(3, dim=-1)

        qkv_out_kv = self.qkv_proj(concatenated_kv_tokens)
        keys = qkv_out_kv[:, :, channels : 2 * channels]
        values = qkv_out_kv[:, :, 2 * channels : 3 * channels]

        batch_windows = batch_size * num_windows_total
        queries = queries.view(
            batch_windows, query_tokens_per_window, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_windows, kv_tokens_per_window, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_windows, kv_tokens_per_window, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_scores = matmul(queries, keys.transpose(-2, -1)) / sqrt(self.head_dim)
        attention_weights = softmax(attention_scores, dim=-1)
        attention_windows = matmul(attention_weights, values)

        attention_windows = (
            attention_windows.transpose(1, 2)
            .contiguous()
            .view(batch_windows, query_tokens_per_window, self.dim)
        )
        attention_windows = self.out_proj(attention_windows)

        attention_windows = attention_windows.view(
            batch_size, num_windows_total, query_tokens_per_window, channels
        )
        attention_windows = attention_windows.view(
            batch_size,
            windows_per_col,
            windows_per_row,
            window_size,
            window_size,
            channels,
        )
        attention_windows = attention_windows.permute(0, 5, 1, 3, 2, 4).contiguous()
        attention_output = attention_windows.view(batch_size, channels, height, width)

        if height_padding > 0 or width_padding > 0:
            attention_output = attention_output[:, :, :original_height, :original_width]

        return attention_output + original_features


class TransformerBlock(Module):
    def __init__(self, dim: int, num_heads: int = 8, win_size: int = 4) -> None:
        assert dim % num_heads == 0, "'dim' must be divisible by 'num_heads'"

        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = win_size

        self.norm1 = LayerNorm(dim)
        self.attn = FocalSelfAttention(dim=dim, num_heads=num_heads, win_size=win_size)
        self.norm2 = LayerNorm(dim)

        self.mlp_fc1 = Linear(dim, 4 * dim)
        self.mlp_act = GELU()
        self.mlp_fc2 = Linear(4 * dim, dim)

    def forward(self, features: Tensor) -> Tensor:
        batch_size, channels, height, width = features.shape

        tokens_before_attn = features.flatten(2).transpose(1, 2)
        normed_tokens_before_attn = self.norm1(tokens_before_attn)
        normed_spatial = normed_tokens_before_attn.transpose(1, 2).view(
            batch_size, channels, height, width
        )

        attn_output_spatial = self.attn(normed_spatial)
        features_after_attn = features + attn_output_spatial

        tokens_after_attn = features_after_attn.flatten(2).transpose(1, 2)
        normed_tokens_after_attn = self.norm2(tokens_after_attn)

        mlp_hidden = self.mlp_fc1(normed_tokens_after_attn)
        mlp_activated = self.mlp_act(mlp_hidden)
        mlp_output = self.mlp_fc2(mlp_activated)
        tokens_after_mlp = tokens_after_attn + mlp_output

        return tokens_after_mlp.transpose(1, 2).view(
            batch_size, channels, height, width
        )


def channels_for_stage(model: Module, stage_idx: int) -> int:
    stage_path = f"stages.{stage_idx}"
    stage = model.get_submodule(stage_path)

    if stage is None:
        raise RuntimeError(f"Stage not found at path: {stage_path}")

    blocks_list = list(stage.get_submodule("blocks").children())

    if len(blocks_list) == 0:
        raise RuntimeError(f"No blocks found under {stage_path}.blocks")

    first_block = blocks_list[0]

    conv_dw = getattr(first_block, "conv_dw", None)

    if conv_dw is not None and hasattr(conv_dw, "in_channels"):
        return int(conv_dw.in_channels)

    norm = getattr(first_block, "norm", None)

    if norm is not None and hasattr(norm, "normalized_shape"):
        ns = norm.normalized_shape

        if isinstance(ns, (tuple, list)) and len(ns) > 0:
            return int(ns[0])

    for m in first_block.modules():
        if isinstance(m, Conv2d) and hasattr(m, "in_channels"):
            return int(m.in_channels)

    raise RuntimeError(
        f"Could not determine channel dim for stage {stage_idx}, inspect your block structure."
    )


def load_model(model_path: Path, with_fsa: bool = False) -> Module:
    device = torch_device("cuda" if torch_cuda_is_available() else "cpu")
    model = build_model(with_fsa=with_fsa)
    model = model.to(device)

    if model_path and model_path.exists():
        model.load_state_dict(torch_load(model_path), strict=True)

    model.eval()

    return model


def build_model(pretrained: bool = False, with_fsa: bool = False) -> Module:
    model = create_model(
        f"convnextv2_atto.fcmae{'_ft_in1k' if pretrained else ''}",
        pretrained=pretrained,
        num_classes=2,
    )

    if pretrained:
        for parameters in model.get_submodule("stem").parameters():
            parameters.requires_grad = False

        for stage_index in [0, 1]:
            for parameters in model.get_submodule(f"stages.{stage_index}").parameters():
                parameters.requires_grad = False

    if with_fsa:
        stage2_blocks = model.get_submodule("stages.2.blocks")
        stage3_blocks = model.get_submodule("stages.3.blocks")

        if stage2_blocks is None or stage3_blocks is None:
            raise RuntimeError("Expected stages.2.blocks and stages.3.blocks in model")

        if not isinstance(stage2_blocks, Sequential) or not isinstance(
            stage3_blocks, Sequential
        ):
            raise RuntimeError("Expected blocks to be Sequential")

        stage2_block_count = len(stage2_blocks)
        stage3_block_count = len(stage3_blocks)

        stage2_channels = channels_for_stage(model, 2)
        stage3_channels = channels_for_stage(model, 3)

        stage2_transformer_blocks = Sequential(
            *[
                TransformerBlock(dim=stage2_channels, num_heads=4, win_size=4)
                for _ in range(stage2_block_count)
            ]
        )
        stage3_transformer_blocks = Sequential(
            *[
                TransformerBlock(dim=stage3_channels, num_heads=8, win_size=4)
                for _ in range(stage3_block_count)
            ]
        )

        model.set_submodule("stages.2.blocks", stage2_transformer_blocks)
        model.set_submodule("stages.3.blocks", stage3_transformer_blocks)

    return model
