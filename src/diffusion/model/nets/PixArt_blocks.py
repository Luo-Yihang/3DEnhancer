# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops
from einops import rearrange
from timm.models.vision_transformer import Mlp, Attention as Attention_


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)

    y = rearrange(y, "b n c -> b c n")

    similarity = x @ y
    return similarity


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **block_kwargs):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentionKVCompress(Attention_):
    """Multi-head Attention block with KV token compression and qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        sampling='conv',
        sr_ratio=1,
        qk_norm=False,
        return_qkv=False,
        use_crossview_module=False,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)

        self.sampling = sampling    # ['conv', 'ave', 'uniform', 'uniform_every']
        self.sr_ratio = sr_ratio
        self.return_qkv = return_qkv
        self.use_crossview_module = use_crossview_module

        if sr_ratio > 1 and sampling == 'conv':
            # Avg Conv Init.
            self.sr = nn.Conv2d(dim, dim, groups=dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr.weight.data.fill_(1/sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = nn.LayerNorm(dim)
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

            self.key_frames_dict = dict()

    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == 'uniform_every':
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == 'ave':
            tensor = F.interpolate(
                tensor, scale_factor=1 / scale_factor, mode='nearest'
            ).permute(0, 2, 3, 1)
        elif sampling == 'uniform':
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == 'conv':
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError

        return tensor.reshape(B, new_N, C).contiguous(), new_N

    def forward(self, x, mask=None, HW=None, block_id=None, qkv_cond=None, n_views=None):
        if self.use_crossview_module:
            # for multi-view row attention
            h = int((x.shape[1])**0.5)
            x = rearrange(x, "(b v) (h w) c -> (b h) (v w) c", v=n_views, h=h)

        B, N, C = x.shape
        if HW is None:
            H = W = int(N ** 0.5)
        else:
            H, W = HW
        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)
    
        new_N = N
        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss

        if qkv_cond is not None:
            assert mask is None
            if use_fp32_attention:
                q, k, v = q.float(), k.float(), v.float()
                qkv_cond = [item.float() for item in qkv_cond]

            v = v + qkv_cond[2]
            attn_bias = None
            x_temp = xformers.ops.memory_efficient_attention(qkv_cond[1], k, v, p=self.attn_drop.p, attn_bias=attn_bias)
            x = xformers.ops.memory_efficient_attention(q, qkv_cond[0], x_temp, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            if use_fp32_attention:
                q, k, v = q.float(), k.float(), v.float()

            attn_bias = None
            if mask is not None:
                attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))

            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, N, C)

        if self.use_crossview_module:
            x = rearrange(x, "(b h) (v w) c -> (b v) (h w) c", v=n_views, h=h)

        x = self.proj(x)
        x = self.proj_drop(x)
        if self.return_qkv:
            return x, [v, k, q]
        else:
            return x

    def forward_with_cross_view(self, x, mask=None, HW=None, block_id=None, qkv_cond=None, epipolar_constrains=None, cam_distances=None, n_views=None):
        B, N, C = x.shape # (b v) (h w) c
        h = int(N**0.5)

        # get multi-view row attention results
        if self.return_qkv:
            x, [v, k, q] = self.forward(x, mask, HW, block_id, qkv_cond, n_views=n_views) # (b v) (h w) c
        else:
            x = self.forward(x, mask, HW, block_id, qkv_cond, n_views=n_views) # (b v) (h w) c

        x = rearrange(x, "(b v) (h w) c -> b v (h w) c", v=n_views, h=h)
        epipolar_constrains = rearrange(epipolar_constrains, "(b v) kv ... -> b v kv ...", v=n_views, kv=2)
        cam_distances = rearrange(cam_distances, "(b v) kv -> b v kv", v=n_views, kv=2)

        # get near-view aggragation results
        x_agg = x.clone()
        for i in range(n_views):
            # near two views are the key views
            kv_idx = [(i-1)%n_views, (i+1)%n_views]

            nv = x_agg[:, [i]] # b 1 (h w) c
            kv = x_agg[:, kv_idx] # b 2 (h w) c

            # sim: b (1 h w) (2 h w)
            with torch.no_grad():
                sim = batch_cosine_sim(
                    rearrange(nv, "b k (h w) c -> b (k h w) c", h=h, k=1),
                    rearrange(kv, "b k (h w) c -> b (k h w) c", h=h, k=2)
                )

                sims = sim.chunk(2, dim=2) # [b 1hw 1hw, b 1hw 1hw]

                idxs = []
                sim_l = []
                for j, sim in enumerate(sims):
                    idx_epipolar = epipolar_constrains[:, i, j] # b hw hw
                    sim[idx_epipolar] = 0
                    sim, sim_idx = sim.max(dim=-1) # b 1hw

                    sim = (sim + 1.) / 2.
                    sim_l.append(((sim)).view(-1, 1 * N, 1).repeat(1, 1, C)) # b 1hw c
                    idxs.append(sim_idx.view(-1, 1 * N, 1).repeat(1, 1, C)) # b 1hw c

                attn_1, attn_2 = kv[:, 0], kv[:, 1]
                attn_output1 = attn_1.gather(dim=1, index=idxs[0]) # b 1hw c
                attn_output2 = attn_2.gather(dim=1, index=idxs[1]) # b 1hw c

                d1 = cam_distances[:, i, 0] # b
                d2 = cam_distances[:, i, 1] # b
                w1 = d2 / (d1 + d2)
                w1 = (w1.unsqueeze(-1).unsqueeze(-1)).to(attn_output1.dtype)

                w1 = (w1 * sim_l[0]) / (w1 * sim_l[0] + (1-w1) * sim_l[1])

                nv_output = w1 * attn_output1 + (1-w1) * attn_output2
                nv_output = rearrange(nv_output, "b (k h w) c -> b k (h w) c", k=1, h=h) # b 1 hw c

            x_agg[:, [i]] = nv + (nv_output - nv).detach()

        x = (x_agg + x) / 2.
        x = rearrange(x, "b v (h w) c -> (b v) (h w) c", v=n_views, h=h)

        if self.return_qkv:
            return x, [v, k, q]
        else:
            return x

    
    def forward_with_cross_view_optimized(self, x, mask=None, HW=None, block_id=None, qkv_cond=None, epipolar_constrains=None, cam_distances=None, n_views=None):
        B, N, C = x.shape # (b v) (h w) c
        h = int(N**0.5)

        # get multi-view row attention results
        if self.return_qkv:
            x, [v, k, q] = self.forward(x, mask, HW, block_id, qkv_cond, n_views=n_views) # (b v) (h w) c
        else:
            x = self.forward(x, mask, HW, block_id, qkv_cond, n_views=n_views) # (b v) (h w) c

        x = rearrange(x, "(b v) (h w) c -> b v (h w) c", v=n_views, h=h)
        epipolar_constrains = rearrange(epipolar_constrains, "(b v) kv ... -> b v kv ...", v=n_views, kv=2)
        cam_distances = rearrange(cam_distances, "(b v) kv -> b v kv", v=n_views, kv=2)

        # get near-view aggragation results
        x_agg = x.clone()
        for i in range(n_views):
            # near two views are the key views
            kv_idx = [(i-1)%n_views, (i+1)%n_views]

            nv = x_agg[:, [i]] # b 1 (h w) c
            kv = x_agg[:, kv_idx] # b 2 (h w) c

            # sim: b (1 h w) (2 h w)
            with torch.no_grad():
                sim = batch_cosine_sim(
                    rearrange(nv, "b k (h w) c -> b (k h w) c", h=h, k=1),
                    rearrange(kv, "b k (h w) c -> b (k h w) c", h=h, k=2)
                )

                sim = sim.chunk(2, dim=2) # [b 1hw 1hw, b 1hw 1hw]
                sim = torch.stack(sim, dim=1) # b 2 hw hw

                idx_epipolar = epipolar_constrains[:, i, :] # b 2 hw hw
                sim[idx_epipolar] = 0

                sim, sim_idx = sim.max(dim=-1) # b 2 hw
                sim = (sim + 1.) / 2.

                sim = sim.unsqueeze(-1).repeat(1, 1, 1, C) # b 2 1hw c
                idx = sim_idx.unsqueeze(-1).repeat(1, 1, 1, C) # b 2 1hw c

                attn_output1 = kv[:, 0].gather(dim=1, index=idx[:, 0]) # b 1hw c
                attn_output2 = kv[:, 1].gather(dim=1, index=idx[:, 1]) # b 1hw c

                d1 = cam_distances[:, i, 0] # b
                d2 = cam_distances[:, i, 1] # b
                w1 = d2 / (d1 + d2)
                w1 = w1.unsqueeze(-1).unsqueeze(-1).to(attn_output1.dtype)
                w1 = (w1 * sim[:, 0]) / (w1 * sim[:, 0] + (1-w1) * sim[:, 1])

                nv_output = w1 * attn_output1 + (1-w1) * attn_output2
                nv_output = rearrange(nv_output, "b (k h w) c -> b k (h w) c", k=1, h=h) # b 1 hw c

            x_agg[:, [i]] = nv + (nv_output - nv).detach()

        x = (x_agg + x) / 2.
        x = rearrange(x, "b v (h w) c -> (b v) (h w) c", v=n_views, h=h)

        if self.return_qkv:
            return x, [v, k, q]
        else:
            return x


#################################################################################
#   AMP attention with fp32 softmax to fix loss NaN problem during training     #
#################################################################################
class Attention(Attention_):
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        use_fp32_attention = getattr(self, 'fp32_attention', False)
        if use_fp32_attention:
            q, k = q.float(), k.float()
        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MaskFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DecoderLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, decoder_hidden_size):
        super().__init__()
        self.norm_decoder = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs//s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class CaptionEmbedderDoubleBr(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.embedding = nn.Parameter(torch.randn(1, in_channels) / 10 ** 0.5)
        self.y_embedding = nn.Parameter(torch.randn(token_num, in_channels) / 10 ** 0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, global_caption, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(global_caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        global_caption = torch.where(drop_ids[:, None], self.embedding, global_caption)
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return global_caption, caption

    def forward(self, caption, train, force_drop_ids=None):
        assert caption.shape[2: ] == self.y_embedding.shape
        global_caption = caption.mean(dim=2).squeeze()
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            global_caption, caption = self.token_drop(global_caption, caption, force_drop_ids)
        y_embed = self.proj(global_caption)
        return y_embed, caption