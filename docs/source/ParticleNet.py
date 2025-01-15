from typing import Optional

import numpy as np
import torch
import torch.nn as nn

"""Adapted from https://github.com/hqucms/weaver/blob/master/utils/nn/model/ParticleNet.py"""


def knn(x, k: int):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][
        :, :, 1:
    ]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k: int, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(
        -1, num_dims
    )  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(
        batch_size, num_points, k, num_dims
    )  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k: int, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(
        num_dims, -1
    )  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(
        num_dims, batch_size, num_points, k
    )  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(
        self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False
    ):
        super().__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.cpu_mode = cpu_mode

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(
                nn.Conv2d(
                    2 * in_feat if i == 0 else out_feats[i - 1],
                    out_feats[i],
                    kernel_size=1,
                    bias=False if self.batch_norm else True,
                )
            )
            self.bns.append(
                nn.BatchNorm2d(out_feats[i]) if batch_norm else nn.Identity()
            )
            self.acts.append(nn.ReLU() if activation else nn.Identity())

        self.sc: nn.Module = nn.Identity()
        self.sc_bn: nn.Module = nn.Identity()
        self.sc_act: nn.Module = nn.Identity()
        if in_feat != out_feats[-1]:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):
        topk_indices = knn(points, self.k)
        x = (
            get_graph_feature_v2(features, self.k, topk_indices)
            if self.cpu_mode
            else get_graph_feature_v1(features, self.k, topk_indices)
        )

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            x = bn(x)
            x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        sc = self.sc(features)  # (N, C_out, P)
        sc = self.sc_bn(sc)

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):
    def __init__(
        self,
        input_dims,
        num_classes,
        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
        fc_params=[(128, 0.1)],
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        for_inference=False,
        for_segmentation=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        self.bn_fts = None
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(
                EdgeConvBlock(
                    k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference
                )
            )

        self.use_fusion = use_fusion
        self.fusion_block = nn.Identity()
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = int(np.clip((in_chn // 128) * 128, 128, 1024))
            self.fusion_block = nn.Sequential(
                nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_chn),
                nn.ReLU(),
            )

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(
                    nn.Sequential(
                        nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(channels),
                        nn.ReLU(),
                        nn.Dropout(drop_rate),
                    )
                )
            else:
                fcs.append(
                    nn.Sequential(
                        nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)
                    )
                )
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask):
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        counts: Optional[torch.Tensor] = None
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        fts = features
        if self.bn_fts is not None and self.use_fts_bn:
            fts = self.bn_fts(fts) * mask

        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

        if self.for_segmentation:
            x = fts
        else:
            if counts is not None and self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output


class FeatureConv(nn.Module):
    def __init__(self, in_chn, out_chn, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ParticleNetTagger(nn.Module):
    def __init__(
        self,
        features_dims,
        num_classes,
        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
        fc_params=[(128, 0.1)],
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        input_dropout=None,
        for_inference=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout else None
        self.conv = FeatureConv(features_dims, 32)
        self.pn = ParticleNet(
            input_dims=32,
            num_classes=num_classes,
            conv_params=conv_params,
            fc_params=fc_params,
            use_fusion=use_fusion,
            use_fts_bn=use_fts_bn,
            use_counts=use_counts,
            for_inference=for_inference,
        )

    def forward(self, points, features, mask):
        if self.input_dropout is not None:
            mask = (self.input_dropout(mask) != 0).float()
            points *= mask
            features *= mask

        points = points
        features = self.conv(features * mask)
        mask = mask

        return self.pn(points, features, mask)
