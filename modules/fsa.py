import math
from typing import TypedDict

import torch
from torch import Tensor
from torch.nn import GELU, AvgPool2d, LayerNorm, Linear, Module


class WindowInfo(TypedDict):
    batch_size: int
    num_windows: int
    height: int
    width: int
    original_height: int
    original_width: int


class WindowPartitioner(Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, features: Tensor) -> tuple[Tensor, WindowInfo]:
        batch, channel, height, width = features.shape

        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size

        if pad_height > 0 or pad_width > 0:
            features = torch.nn.functional.pad(features, (0, pad_width, 0, pad_height))
            height, width = features.shape[2], features.shape[3]

        windows = torch.nn.functional.unfold(
            features, kernel_size=self.window_size, stride=self.window_size
        )

        num_windows = (height // self.window_size) * (width // self.window_size)

        tokens = (
            windows.transpose(1, 2)
            .reshape(batch * num_windows, channel, self.window_size * self.window_size)
            .transpose(1, 2)
        )

        return tokens, {
            "batch_size": batch,
            "num_windows": num_windows,
            "height": height,
            "width": width,
            "original_height": height - pad_height,
            "original_width": width - pad_width,
        }


class LocalContextExtractor(Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.local_size = 2 * window_size

    def forward(self, features: Tensor, num_windows: int) -> Tensor:
        batch, channel, height, width = features.shape

        target_size = (
            int(math.sqrt(num_windows)) - 1
        ) * self.window_size + self.local_size

        pad_height = max(0, target_size - height)
        pad_width = max(0, target_size - width)

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        features = torch.nn.functional.pad(
            features, (pad_left, pad_right, pad_top, pad_bottom)
        )

        return (
            torch.nn.functional.unfold(
                features, kernel_size=self.local_size, stride=self.window_size
            )
            .view(batch, channel, self.local_size * self.local_size, -1)
            .permute(0, 3, 2, 1)
            .contiguous()
            .view(batch * num_windows, self.local_size * self.local_size, channel)
        )


class MultiScalePooler(Module):
    def __init__(self) -> None:
        super().__init__()
        self.mid_pool = AvgPool2d(kernel_size=2, stride=2)
        self.global_pool = AvgPool2d(kernel_size=4, stride=4)

    def forward(self, features: Tensor, num_windows: int) -> tuple[Tensor, Tensor]:
        batch, channel, height, width = features.shape

        mid_tokens = self.mid_pool(features).flatten(2).transpose(1, 2)

        mid_tokens = (
            mid_tokens.unsqueeze(1)
            .expand(batch, num_windows, mid_tokens.shape[1], channel)
            .reshape(batch * num_windows, -1, channel)
        )

        global_tokens = self.global_pool(features).flatten(2).transpose(1, 2)

        global_tokens = (
            global_tokens.unsqueeze(1)
            .expand(batch, num_windows, global_tokens.shape[1], channel)
            .reshape(batch * num_windows, -1, channel)
        )

        return mid_tokens, global_tokens


class MultiHeadAttention(Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
        batch, num_q, channel = queries.shape
        num_kv = keys.shape[1]

        query = queries.view(batch, num_q, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = keys.view(batch, num_kv, self.num_heads, self.head_dim).transpose(1, 2)
        value = values.view(batch, num_kv, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)

        return (
            torch.matmul(weights, value)
            .transpose(1, 2)
            .contiguous()
            .view(batch, num_q, channel)
        )


class WindowReconstructor(Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, tokens: Tensor, info: WindowInfo) -> Tensor:
        batch = info["batch_size"]
        num_windows = info["num_windows"]
        height, width = info["height"], info["width"]
        channel = tokens.shape[-1]

        windows_height = height // self.window_size
        windows_width = width // self.window_size

        tokens = tokens.view(
            batch, num_windows, self.window_size * self.window_size, channel
        )

        tokens = tokens.view(
            batch,
            windows_height,
            windows_width,
            self.window_size,
            self.window_size,
            channel,
        )

        tokens = tokens.permute(0, 5, 1, 3, 2, 4).contiguous()
        output = tokens.view(batch, channel, height, width)

        if info["original_height"] < height or info["original_width"] < width:
            output = output[:, :, : info["original_height"], : info["original_width"]]

        return output


class FocalSelfAttention(Module):
    def __init__(self, dim: int, num_heads: int = 8, win_size: int = 4) -> None:
        super().__init__()
        self.dim = dim

        self.window_partition = WindowPartitioner(win_size)
        self.local_context = LocalContextExtractor(win_size)
        self.multi_scale = MultiScalePooler()

        self.token_norm = LayerNorm(dim)
        self.q_projection = Linear(dim, dim)
        self.kv_projection = Linear(dim, dim * 2)

        self.attention = MultiHeadAttention(dim, num_heads)
        self.out_projection = Linear(dim, dim)

    def forward(self, features: Tensor) -> Tensor:
        residual = features

        query_tokens, info = self.window_partition(features)
        query_tokens: Tensor = self.token_norm(query_tokens)

        local_tokens: Tensor = self.local_context(features, info["num_windows"])
        mid_tokens, global_tokens = self.multi_scale(features, info["num_windows"])

        kv_tokens = torch.cat([local_tokens, mid_tokens, global_tokens], dim=1)

        queries: Tensor = self.q_projection(query_tokens)
        kv: Tensor = self.kv_projection(kv_tokens)
        keys, values = kv.chunk(2, dim=-1)

        attended: Tensor = self.attention(queries, keys, values)
        output: Tensor = self.out_projection(attended)

        reconstructor = WindowReconstructor(self.window_partition.window_size)
        output: Tensor = reconstructor(output, info)

        return output + residual


class TransformerBlock(Module):
    def __init__(self, dim: int, num_heads: int = 8, win_size: int = 4) -> None:
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = FocalSelfAttention(dim, num_heads, win_size)

        self.norm2 = LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            Linear(dim, 4 * dim),
            GELU(),
            Linear(4 * dim, dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        batch, channel, height, width = features.shape

        normed = features.flatten(2).transpose(1, 2)
        normed: Tensor = self.norm1(normed)
        normed = normed.transpose(1, 2).view(batch, channel, height, width)

        features: Tensor = self.attn(normed)

        tokens = features.flatten(2).transpose(1, 2)
        normed: Tensor = self.norm2(tokens)
        tokens: Tensor = tokens + self.mlp(normed)

        return tokens.transpose(1, 2).view(batch, channel, height, width)
