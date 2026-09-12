# encoding: utf-8
"""
VeRi SBS R50-ibn 车辆 ReID 模型的**自实现**版本（纯 torch / torchvision）。

为什么会有这个文件：
    fast-reid (JDAI-CV) 没有 setup.py，官方用法是把源码树挂到 sys.path 再 import。
    本仓库的运行环境对该第三方源码树的执行有额外限制，因此这里**不改动 fast-reid 源码**，
    而是照其源码结构重写一份等价网络，只从 release 权重文件里读取张量
    （torch.load(..., weights_only=True)，不执行 pickle 里的任意代码）。

    结构完全对齐以下 fast-reid 文件（本文件注释里给出对应位置，便于人工核对）：
      - fastreid/modeling/backbones/resnet.py   : Bottleneck / ResNet / _build_nonlocal 的 NL 位置
      - fastreid/layers/non_local.py            : Non_local
      - fastreid/layers/batch_norm.py           : IBN, BatchNorm(get_norm('BN'))
      - fastreid/layers/pooling.py              : GeneralizedMeanPoolingP
      - fastreid/modeling/heads/embedding_head.py: EmbeddingHead(pool -> bottleneck(BN) -> [...,0,0])
      - fastreid/modeling/meta_arch/baseline.py : preprocess_image = (x - pixel_mean) / pixel_std

    配置来源 configs/VeRi/sbs_R50-ibn.yml + configs/Base-SBS.yml + configs/Base-bagtricks.yml：
      DEPTH=50x, LAST_STRIDE=1, WITH_IBN=True, WITH_NL=True, NORM=BN,
      HEADS.POOL_LAYER=GeneralizedMeanPoolingP, HEADS.WITH_BNNECK=True, HEADS.NECK_FEAT=after,
      INPUT.SIZE_TEST=[256,256], PIXEL_MEAN/STD = ImageNet*255（实际以权重内的 buffer 为准）

键名对齐：
    权重里 heads 下是 `heads.bottleneck.0.*`（BN2d 2048）+ `heads.classifier.weight`(575,2048)，
    外加一个孤立的 `heads.bnneck.num_batches_tracked`（早期版本 `bnneck` 改名时漏掉的历史遗留，
    推理无用）。因此本文件把颈部 BN 命名为 `bottleneck.0`、分类层命名为 `classifier`，
    以便 load_state_dict(strict=True) 能直接通过。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VeriReID", "load_veri_reid"]

# configs/VeRi/sbs_R50-ibn.yml -> INPUT.SIZE_TEST
SIZE_TEST = (256, 256)
FEAT_DIM = 2048
# 权重里 heads.classifier.weight 的第 0 维（VeRi 训练集类别数）；推理不用，仅用于建层
NUM_CLASSES = 575
# num_blocks_per_stage / nl_layers_per_stage for '50x' (resnet.py build_resnet_backbone)
BLOCKS_50X = [3, 4, 6, 3]
# 与权重中 backbone.NL_2.* (2 个) / backbone.NL_3.* (3 个) 实测一致
NL_LAYERS_50X = [0, 2, 3, 0]


class BatchNorm(nn.BatchNorm2d):
    """fastreid/layers/batch_norm.py BatchNorm（默认 eps=1e-5, momentum=0.1）。"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, bias_freeze=False, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.bias.requires_grad_(not bias_freeze)


class IBN(nn.Module):
    """fastreid/layers/batch_norm.py IBN：前半通道 InstanceNorm，后半通道 BatchNorm。"""

    def __init__(self, planes):
        super().__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = BatchNorm(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), 1)


class NonLocal(nn.Module):
    """fastreid/layers/non_local.py Non_local（reduc_ratio=2 -> inter_channels = 1）。"""

    def __init__(self, in_channels, reduc_ratio=2):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio   # == 1，与官方实现一致
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1, 1, 0)
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1, 1, 0),
            BatchNorm(in_channels),
        )
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1, 1, 0)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f / f.size(-1)
        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        return self.W(y) + x


class Bottleneck(nn.Module):
    """fastreid/modeling/backbones/resnet.py Bottleneck（expansion=4）。"""

    expansion = 4

    def __init__(self, inplanes, planes, with_ibn=False, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if with_ibn else BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNetBackbone(nn.Module):
    """ResNet50 + IBN(layer1/2/3) + Non-Local(layer2 末两个 block、layer3 末三个 block 之后)。

    注意 resnet.py 里 layer4 的 _make_layer **没有**传 with_ibn（只有 layer1/2/3 传了），
    实测权重也是 layer1/2/3 有 IN、layer4 没有，与此一致。
    """

    def __init__(self, last_stride=1, blocks=None, nl_layers=None):
        super().__init__()
        blocks = blocks or BLOCKS_50X
        nl_layers = nl_layers or NL_LAYERS_50X

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        # 官方用的是 ceil_mode=True（不是标准的 MaxPool2d 默认）
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(64, blocks[0], 1, with_ibn=True)
        self.layer2 = self._make_layer(128, blocks[1], 2, with_ibn=True)
        self.layer3 = self._make_layer(256, blocks[2], 2, with_ibn=True)
        self.layer4 = self._make_layer(512, blocks[3], last_stride, with_ibn=False)

        # _build_nonlocal: NL_1/NL_4 数量为 0，NL_2/NL_3 挂在各自 stage 的**最后** n 个 block 之后
        # NL_x_idx = sorted([layers[x] - (i+1) for i in range(nl_layers[x])])
        self.NL_1_idx = sorted([blocks[0] - (i + 1) for i in range(nl_layers[0])])
        self.NL_2_idx = sorted([blocks[1] - (i + 1) for i in range(nl_layers[1])])
        self.NL_3_idx = sorted([blocks[2] - (i + 1) for i in range(nl_layers[2])])
        self.NL_4_idx = sorted([blocks[3] - (i + 1) for i in range(nl_layers[3])])
        self.NL_1 = nn.ModuleList([NonLocal(256) for _ in range(nl_layers[0])])
        self.NL_2 = nn.ModuleList([NonLocal(512) for _ in range(nl_layers[1])])
        self.NL_3 = nn.ModuleList([NonLocal(1024) for _ in range(nl_layers[2])])
        self.NL_4 = nn.ModuleList([NonLocal(2048) for _ in range(nl_layers[3])])

    def _make_layer(self, planes, num_blocks, stride, with_ibn):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, with_ibn, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.inplanes, planes, with_ibn))
        return nn.Sequential(*layers)

    @staticmethod
    def _run_stage(stage, nl_modules, nl_idx, x):
        """跑一个 stage，并在指定下标之后插入 Non-Local 模块。"""
        counter = 0
        for i in range(len(stage)):
            x = stage[i](x)
            if counter < len(nl_idx) and i == nl_idx[counter]:
                x = nl_modules[counter](x)
                counter += 1
        return x

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self._run_stage(self.layer1, self.NL_1, self.NL_1_idx, x)
        x = self._run_stage(self.layer2, self.NL_2, self.NL_2_idx, x)
        x = self._run_stage(self.layer3, self.NL_3, self.NL_3_idx, x)
        x = self._run_stage(self.layer4, self.NL_4, self.NL_4_idx, x)
        return x


class GeneralizedMeanPoolingP(nn.Module):
    """fastreid/layers/pooling.py GeneralizedMeanPoolingP（p 可学习，默认 3）。"""

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1.0 / self.p)


class EmbeddingHead(nn.Module):
    """fastreid/modeling/heads/embedding_head.py EmbeddingHead。

    POOL_LAYER=GeneralizedMeanPoolingP, WITH_BNNECK=True, EMBEDDING_DIM=0(无降维 conv),
    NECK_FEAT=after。eval 模式下 forward 直接返回 neck 特征（即 BN 之后）。
    """

    def __init__(self, feat_dim=FEAT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool_layer = GeneralizedMeanPoolingP()
        self.bottleneck = nn.Sequential(BatchNorm(feat_dim, bias_freeze=True))
        # 仅训练用；命名与权重里的 heads.classifier.weight 对齐
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, features):
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        return neck_feat[..., 0, 0]


class VeriReID(nn.Module):
    """完整推理模型：像素归一化 -> backbone -> embedding head。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1), False)
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1), False)
        self.backbone = ResNetBackbone()
        self.heads = EmbeddingHead()

    @torch.no_grad()
    def forward(self, images):
        """images: float32 tensor (B,3,H,W)，**未归一化**的 0-255 RGB（与官方 demo 一致）。"""
        x = (images - self.pixel_mean) / self.pixel_std
        feats = self.backbone(x)
        feats = self.heads(feats)
        return F.normalize(feats, dim=1)


# 权重中 `heads.bnneck.num_batches_tracked` 是早期版本 bnneck 改名后遗留的孤立 buffer，
# 与推理无关；strict 加载时显式丢弃。
_IGNORED_KEYS = ("heads.bnneck.num_batches_tracked",)


def load_veri_reid(weights_path, device="cuda"):
    """构建自实现模型并加载官方权重，返回 (model, report)。

    只读取权重张量本身（weights_only=True），不执行权重文件里的任何代码。
    """
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model = VeriReID()
    model_sd = model.state_dict()

    dropped = [k for k in state if k in _IGNORED_KEYS]
    filtered = {k: v for k, v in state.items() if k not in _IGNORED_KEYS}

    missing = [k for k in model_sd if k not in filtered]
    unexpected = [k for k in filtered if k not in model_sd]
    shape_mismatch = [
        (k, tuple(filtered[k].shape), tuple(model_sd[k].shape))
        for k in filtered if k in model_sd and tuple(filtered[k].shape) != tuple(model_sd[k].shape)
    ]

    report = {
        "checkpoint_keys": len(state),
        "dropped_keys": dropped,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "pixel_mean": filtered["pixel_mean"].flatten().tolist() if "pixel_mean" in filtered else None,
        "pixel_std": filtered["pixel_std"].flatten().tolist() if "pixel_std" in filtered else None,
    }

    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "自实现结构与权重不匹配，拒绝继续（避免静默产出错误特征）：\n"
            f"  missing       : {missing[:10]} (共 {len(missing)})\n"
            f"  unexpected    : {unexpected[:10]} (共 {len(unexpected)})\n"
            f"  shape_mismatch: {shape_mismatch[:10]} (共 {len(shape_mismatch)})"
        )

    model.load_state_dict(filtered, strict=True)
    model.eval()
    model.to(device)
    report["loaded_keys"] = len(filtered)
    return model, report
