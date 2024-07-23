import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat
import math

from causal_conv1d import causal_conv1d_fn
import selective_scan_cuda

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd()
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


class MotionAwareStateSpaceModel(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_rank="auto", conv_bias=True, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.conv_bias = conv_bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dim = d_model
        self.d_inner = int(self.expand * self.d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.multi_act = nn.Sigmoid()

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, **self.factory_kwargs)
        # self.in_proj_single = nn.Linear(self.d_model, self.d_inner, bias=False, **self.factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv // 2,
            **self.factory_kwargs,
        )
        self.conv1d_single = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv // 2,
            **self.factory_kwargs,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **self.factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **self.factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **self.factory_kwargs)

        self.dt_init()
        self.A_init()
        self.D_init()

    def A_init(self):
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.device), "n -> d n", d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

    def dt_init(self, dt_init="random", dt_scale=1.0, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **self.factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

    def D_init(self):
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=self.device))  # Keep in fp32
        self.D._no_weight_decay = True

    def ssm(self, conv1d_out, z,
            x_proj_weight, delta_proj_weight,
            A, B=None, C=None, D=None,
            delta_bias=None, B_proj_bias=None, C_proj_bias=None, delta_softplus=True):
        L = conv1d_out.shape[-1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        delta_rank = self.dt_proj.weight.shape[1]

        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)

        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None: D = D.contiguous()

        out = selective_scan_fn(conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus)

        # return F.linear(rearrange(out, "b d l -> b l d"), out_proj_weight, out_proj_bias)
        return out

    def single_transform(self, single_feat, b_bincount, single_mask, multi_mask, tn_inverse, multi_seqlen):
        B = b_bincount.shape[0]
        max_bincount = torch.max(b_bincount)
        BT, D, BT_L = single_feat.shape
        single_feat = rearrange(single_feat, "b d l -> b l d").reshape(BT*BT_L, D)

        # pad inverse
        # pad_inverse_feat = []
        # for i in range(BT):
        #     this_batch_feat = single_feat[BT_L * i: BT_L * (i + 1), :]
        #     pad_inverse_feat.append(this_batch_feat[:bt_bincount[i], :])
        # single_feat = torch.concat(pad_inverse_feat)

        # fast pad inverse
        # single pad inverse
        single_feat = single_feat[single_mask]

        # serialized inverse
        single_feat = single_feat[tn_inverse]

        # multi pad
        single_feat_pad = torch.zeros(B * max_bincount, D, device=single_feat.device)
        single_feat_pad[multi_mask] = single_feat

        # single_feat_pad = []
        # for b in range(B):
        #     bn_mask = point_batch == b
        #     single_feat_pad.append(torch.nn.functional.pad(single_feat[bn_mask].T, (0, max_bincount - b_bincount[b]), 'constant', 0).T)
        # single_feat_pad = torch.concat(single_feat_pad)

        return rearrange(single_feat_pad.reshape(B, multi_seqlen, D), "b l d -> b d l")

    @autocast(enabled=False)
    def forward(self, multi_x, single_x, multi_mask, single_mask, b_bincount, tn_inverse):
        if multi_x.dtype == torch.float16:
            multi_x = multi_x.type(torch.float32)
        if single_x.dtype == torch.float16:
            single_x = single_x.type(torch.float32)
        multi_batch, multi_seqlen, C = multi_x.shape
        single_batch, single_seqlen, _ = single_x.shape
        assert C == self.d_model

        multi_hidden_states = self.norm(multi_x)
        single_hidden_states = self.norm(single_x)

        # rewrite ssm
        # x_mamba = self.mamba(x_norm)

        # We do matmul and transpose BLH -> HBL at the same time
        multix_z = rearrange(
            self.in_proj.weight @ rearrange(multi_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=multi_seqlen,
        )
        single_x = rearrange(
            self.in_proj.weight[:self.d_inner, :] @ rearrange(single_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=single_seqlen,
        )

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        if multix_z.stride(-1) != 1:
            multix_z = multix_z.contiguous()
        if single_x.stride(-1) != 1:
            single_x = single_x.contiguous()
        multix, z = multix_z.chunk(2, dim=1)

        multi_conv1d_out = self.conv1d(multix)
        multi_conv1d_out_weight = self.multi_act(multi_conv1d_out)
        single_feat = self.act(self.conv1d_single(single_x))

        # t_log = time.time()
        # single feat transform
        single_feat = self.single_transform(single_feat=single_feat, b_bincount=b_bincount,
                                            single_mask=single_mask, multi_mask=multi_mask,
                                            tn_inverse=tn_inverse, multi_seqlen=multi_seqlen)
        # print("1 cost {:.3f}s".format(time.time() - t_log))

        # t_log = time.time()
        mgafeat = self.act(torch.mul(multi_conv1d_out_weight, single_feat) + multi_conv1d_out)
        # print("2 cost {:.3f}s".format(time.time() - t_log))

        ssm_out = self.ssm(conv1d_out=mgafeat, z=z,
                           x_proj_weight=self.x_proj.weight, delta_proj_weight=self.dt_proj.weight,
                           A=A, B=None, C=None, D=self.D.float(),
                           delta_bias=self.dt_proj.bias.float(), delta_softplus=True)

        mamba_out = self.out_proj(rearrange(ssm_out, "b d l -> b l d"))

        return mamba_out