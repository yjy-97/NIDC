import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F
from Aggregation import MGC
from Interaction import SelfAttention

torch.autograd.set_detect_anomaly(True)

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class PointRecuder(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = self.to_3tuple(patch_size)
        stride = self.to_3tuple(stride)
        padding = self.to_3tuple(padding)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def to_3tuple(self, value):
        if isinstance(value, int):
            return (value, value, value)
        else:
            return to_2tuple(value)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, proposal_d=2, fold_w=2, fold_h=2, fold_d=2, heads=4,
                 head_dim=24,
                 return_center=False):

        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv3d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv3d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv3d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool3d((proposal_w, proposal_h, proposal_d))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.fold_d = fold_d
        self.return_center = return_center
        self.temperature = 0.5
        self.alp = 0.2
        self.tanh = nn.Tanh()
        self.sa = SelfAttention(32)
        self.mgc = MGC()

    def forward(self, x):
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h d-> (b e) c w h d", e=self.heads)
        value = rearrange(value, "b (e c) w h d-> (b e) c w h d", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0, d0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) (f3 d)-> (b f1 f2 f3) c w h d", f1=self.fold_w,
                          f2=self.fold_h, f3=self.fold_d)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) (f3 d)-> (b f1 f2 f3) c w h d", f1=self.fold_w, f2=self.fold_h,
                              f3=self.fold_d)
        b, c, w, h, d = x.shape
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), 'b c w h d-> b (w h d) c')
        b, c, ww, hh, dd = centers.shape
        # sim = torch.sigmoid(
        #     self.sim_beta +
        #     self.sim_alpha * (pairwise_cos_sim(
        #         centers.reshape(b, c, -1).permute(0, 2, 1),
        #         x.reshape(b, c, -1).permute(0, 2, 1)
        #     ) + self.alp * self.tanh(torch.sqrt(torch.cdist(centers.reshape(b, c, -1).permute(0, 2, 1),
        #                                                     x.reshape(b, c, -1).permute(0, 2, 1)))))
        # )  # [B,M,N]
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * (pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            ))
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)

        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h d-> b (w h d) c')  # [B,N,D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        out = self.mgc(out)
        out = self.sa(out)

        if self.return_center:
            out = rearrange(out, "b (w h d) c-> b c w h d", w=ww, h=hh)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h d) c-> b c w h d", w=w, h=h)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2 f3) c w h d-> b c (f1 w) (f2 h) (f3 d)", f1=self.fold_w, f2=self.fold_h,
                            f3=self.fold_d)
        out = rearrange(out, "(b e) c w h d-> b (e c) w h d", e=self.heads)
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 4, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 4, 1, 2, 3)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, fold_d=2, heads=4, head_dim=24, return_center=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, fold_d=fold_d, heads=heads, head_dim=head_dim,
                                   return_center=False)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:

            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, fold_d=2, heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
            heads=heads, head_dim=head_dim, return_center=False
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


class ICC(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=2,
                 in_patch_size=4, in_stride=4, in_pad=0,
                 down_patch_size=2, down_stride=2, down_pad=0,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 fold_d=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32],
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PointRecuder(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=4, embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                 fold_w=fold_w[i], fold_h=fold_h[i], fold_d=fold_d[i], heads=heads[i],
                                 head_dim=head_dim[i],
                                 return_center=False
                                 )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PointRecuder(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
        self.head1 = nn.Linear(4, 2)
        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

        self.fc = nn.Linear(256, 2)
        self.tanh = nn.Tanh()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x, tem):
        _, c, img_w, img_h, img_d = x.shape
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        range_d = torch.arange(0, img_d, step=1) / (img_w - 1.0)  # 添加range_d
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, range_d), dim=-1).float()  # 使用torch.meshgrid生成三维网格
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5

        pos = fea_pos.permute(3, 0, 1, 2).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1, -1)
        tem = tem[:-1, 12:-13, :-1].unsqueeze(0).repeat(pos.shape[0], 1, 1, 1).cuda(2)
        tem = tem.unsqueeze(1).repeat(1, 3, 1, 1, 1).cuda(2)
        pos = pos.cuda(2) + tem.cuda(2)
        x = self.patch_embed(torch.cat([x, pos], dim=1))
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x, tem):
        x = self.forward_embeddings(x, tem)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x

        x = self.norm(x)
        F = x.view(x.shape[0], -1)
        F = self.tanh(F)

        cls_out = self.fc(x.view(x.shape[0], -1))
        return cls_out, F


@register_model
def NIDC_base_dim64(pretrained=False, **kwargs):
    layers = [1, 1, 3, 1]
    norm_layer = GroupNorm
    embed_dims = [16, 32, 64, 4]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [4, 2, 1, 1]
    fold_h = [4, 2, 1, 1]
    fold_d = [4, 2, 1, 1]
    heads = [2, 2, 2, 2]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ICC(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h, fold_d=fold_d,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


if __name__ == '__main__':
    input = torch.rand(32, 1, 112, 112, 112)
    input1 = torch.rand(113, 137, 113)
    model = coc_base_dim64()
    out = model(input, input1)
    print(out.shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters / 1024 ** 2))
