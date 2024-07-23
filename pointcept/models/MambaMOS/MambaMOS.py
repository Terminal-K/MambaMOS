from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential

from torch.cuda.amp import autocast
from mamba_ssm.modules.mamba_simple import Mamba
from .mssm import MotionAwareStateSpaceModel

class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, n_tokens, C = x.shape
        assert C == self.dim

        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)

        return x_mamba

class MotionAwareStateSpaceModelBlock(PointModule):
    def __init__(
        self,
        channels,
        gather_num,
        order_index=0,
    ):
        super().__init__()
        self.channels = channels
        self.order_index = order_index

        self.gather_num = gather_num

        # self.mamba = MambaLayer(dim=channels)
        self.mamba = MotionAwareStateSpaceModel(d_model=channels, d_conv=5)

    @torch.no_grad()
    def get_padding(self, point):
        offset = point.offset
        bt_bincount = torch.bincount(point.bt)
        b_bincount = offset2bincount(offset)
        B = b_bincount.shape[0]
        BT = bt_bincount.shape[0]

        tn_max_bincount = torch.max(bt_bincount)
        max_bincount = torch.max(b_bincount).item()

        max_indices = b_bincount.unsqueeze(1)
        indices = torch.arange(max_bincount, device=b_bincount.device).unsqueeze(0).expand(B, -1)
        multi_mask = indices < max_indices
        multi_mask = multi_mask.view(-1)

        single_max_indices = bt_bincount.unsqueeze(1)
        single_indices = torch.arange(tn_max_bincount, device=bt_bincount.device).unsqueeze(0).expand(BT, -1)
        single_mask = single_indices < single_max_indices
        single_mask = single_mask.view(-1)

        return multi_mask, single_mask

    def get_inverse(self, feat, B, L, mask):
        # reverse
        feat = feat.reshape(B * L, -1)
        reverse_feat = feat[mask.view(-1)]
        return reverse_feat

    @torch.no_grad()
    def serialized(self, point):
        sp_inds = point.serialized_order[self.order_index, :]
        sp_inverse = point.serialized_inverse[self.order_index, :]

        n = torch.bincount(point.bt)
        tn_inds = []

        space_bt = point.bt[sp_inds]
        for i in range(len(n)):
            tn_inds.append(torch.nonzero(space_bt == i).squeeze(1))
        sptn_inds = torch.concat(tn_inds)
        sptn_inverse = torch.argsort(sptn_inds)
        return sp_inds, sp_inverse, sptn_inds, sptn_inverse

    def ori_mamba_forward(self, point):
        B = point.offset.shape[0]
        C = point.feat.shape[1]
        b_bincount = offset2bincount(point.offset)
        max_bincount = torch.max(b_bincount)
        tn_max_bincount = torch.max(torch.bincount(point.bt))

        # serialized
        sp_inds, sp_inverse, tn_inds, tn_inverse = self.serialized(point)
        multi_serialized_feat = point.feat[sp_inds]

        # pad
        multi_feat_pad = torch.zeros(B * max_bincount, C, dtype=point.feat.dtype, device=multi_serialized_feat.device)
        multi_mask, single_mask = self.get_padding(point)
        multi_feat_pad[multi_mask] = multi_serialized_feat

        # reshape
        L, BT_L = max_bincount, tn_max_bincount
        multi_feat_pad = multi_feat_pad.reshape(B, L, -1)

        feat = self.mamba(multi_feat_pad)

        # pad inverse
        point.feat = self.get_inverse(feat, B, L, multi_mask)

        # space inverse
        point.feat = point.feat[sp_inverse]  # L, D

        return point

    def mssm_forward(self, point):
        B = point.offset.shape[0]
        BT = B * self.gather_num
        C = point.feat.shape[1]
        b_bincount = offset2bincount(point.offset)
        max_bincount = torch.max(b_bincount)
        tn_max_bincount = torch.max(torch.bincount(point.bt))

        # serialized
        sp_inds, sp_inverse, tn_inds, tn_inverse = self.serialized(point)
        multi_serialized_feat = point.feat[sp_inds]
        single_serialized_feat = point.feat[sp_inds[tn_inds]]

        # pad
        multi_feat_pad = torch.zeros(B * max_bincount, C, dtype=point.feat.dtype, device=multi_serialized_feat.device)
        single_feat_pad = torch.zeros(BT * tn_max_bincount, C, dtype=point.feat.dtype, device=single_serialized_feat.device)
        multi_mask, single_mask = self.get_padding(point)
        multi_feat_pad[multi_mask] = multi_serialized_feat
        single_feat_pad[single_mask] = single_serialized_feat

        # reshape
        L, BT_L = max_bincount, tn_max_bincount
        multi_feat_pad = multi_feat_pad.reshape(B, L, -1)
        single_feat_pad = single_feat_pad.reshape(BT, BT_L, -1)

        feat = self.mamba(multi_feat_pad, single_feat_pad, multi_mask, single_mask, b_bincount, tn_inverse) # B L C
        # feat = self.mamba(multi_feat_pad)

        # pad inverse
        point.feat = self.get_inverse(feat, B, L, multi_mask)

        # space inverse
        point.feat = point.feat[sp_inverse]        # L, D

        return point

    def forward(self, point):
        # return self.ori_mamba_forward(point)
        return self.mssm_forward(point)

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(PointModule):
    def __init__(
        self,
        channels,
        gather_num,
        mlp_ratio=4.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = MotionAwareStateSpaceModelBlock(
            channels=channels,
            gather_num=gather_num,
            order_index=order_index,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.norm1(point)

        point = self.drop_path(self.attn(point))

        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            segment=point.segment[head_indices],
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
            tn=point.tn[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point

class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if "odl_loss" in point.keys():
            parent.odl_loss = point.odl_loss

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent

class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point

class TCBE(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.seq = PointSequential()
        # TODO: check remove spconv
        self.stem = nn.Sequential(nn.Conv1d(in_channels-1, embed_channels, kernel_size=3, padding=1, bias=False),
                                  act_layer())
        self.tn_stem1 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
                                      act_layer())
        self.tn_stem2 = nn.Sequential(nn.Conv1d(1, embed_channels, kernel_size=3, padding=1, bias=False),
                                      act_layer())
        self.stem_final = nn.Sequential(nn.Conv1d(embed_channels, embed_channels, kernel_size=3, padding=1, bias=False),
                                        norm_layer(embed_channels),
                                        act_layer())

        self.drop = nn.Dropout(0.3)

    @torch.no_grad()
    def serialized(self, point):
        sp_inds = point.serialized_order[0, :]
        sp_inverse = point.serialized_inverse[0, :]

        return sp_inds, sp_inverse

    @torch.no_grad()
    def get_padding(self, point):
        offset = point.offset
        b_bincount = offset2bincount(offset)
        B = b_bincount.shape[0]

        max_bincount = torch.max(b_bincount).item()

        max_indices = b_bincount.unsqueeze(1)
        indices = torch.arange(max_bincount, device=b_bincount.device).unsqueeze(0).expand(B, -1)
        multi_mask = indices < max_indices
        multi_mask = multi_mask.view(-1)

        return multi_mask

    def get_inverse(self, feat, B, L, mask):
        # reverse
        feat = feat.reshape(B * L, -1)
        reverse_feat = feat[mask.view(-1)]
        return reverse_feat

    def forward(self, point: Point, point_tn: Point):
        sp_inds, sp_inverse = self.serialized(point)
        feat = point.sparse_conv_feat.features[sp_inds]
        tn_feat = point_tn.sparse_conv_feat.features[sp_inds]

        B = point.offset.shape[0]
        b_bincount = offset2bincount(point.offset)
        max_bincount = torch.max(b_bincount)
        pad_feat = torch.zeros(B * max_bincount, self.in_channels-1, device=feat.device)
        pad_tn_feat = torch.zeros(B * max_bincount, 1, device=feat.device)
        mask = self.get_padding(point)
        pad_feat[mask] = feat
        pad_tn_feat[mask] = tn_feat
        pad_feat = pad_feat.reshape(B, max_bincount, -1).permute(0, 2, 1)
        pad_tn_feat = pad_tn_feat.reshape(B, max_bincount, -1).permute(0, 2, 1)

        pad_feat = self.stem(pad_feat)
        tn1_sparse_feat = self.tn_stem1(pad_tn_feat)
        tn2_sparse_feat = self.tn_stem2(pad_tn_feat)

        pad_feat = pad_feat + tn2_sparse_feat + pad_feat * tn1_sparse_feat
        pad_feat = self.stem_final(pad_feat)
        pad_feat = self.drop(pad_feat)

        pad_feat = self.get_inverse(pad_feat.permute(0, 2, 1), B, max_bincount, mask)

        pad_feat = pad_feat[sp_inverse]
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(pad_feat)
        point.feat = pad_feat

        return point

@MODELS.register_module("MambaMOS")
class MambaMOS(PointModule):
    def __init__(
        self,
        in_channels=6,
        gather_num=4,
        order=("z", "z_trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        mlp_ratio=4,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = TCBE(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        gather_num=gather_num,
                        mlp_ratio=mlp_ratio,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            gather_num=gather_num,
                            mlp_ratio=mlp_ratio,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def get_tn_sparse_feat(self, point):
        sparse_conv_tn_feat = spconv.SparseConvTensor(
            features=point.feat[:, -1:] + 1,
            indices=torch.cat(
                [point.batch.unsqueeze(-1).int(), point.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=point.sparse_conv_feat.spatial_shape,
            batch_size=point.batch[-1].tolist() + 1,
        )
        return Point({"sparse_conv_feat": sparse_conv_tn_feat})

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify(wtn=False)
        point_tn = self.get_tn_sparse_feat(point)

        point = self.embedding(point, point_tn)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )

        return point
