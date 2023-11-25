import time
import torch
from torch import nn
import torchvision
import numpy as np
import pkg_resources as pkg


def _check_version(
    current: str = "0.0.0",
    minimum: str = "0.0.0",
    name: str = "version ",
    pinned: bool = False,
    hard: bool = False,
    verbose: bool = False,
) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    warning_message = f"WARNING ⚠️ {name}{minimum} is required by YOLOv8, but {name}{current} is currently installed"
    if hard:
        assert result, warning_message  # assert min requirements met
    if verbose and not result:
        print(warning_message)
    return result


TORCH_1_10 = _check_version(torch.__version__, "1.10.0")


def _make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = (
            torch.meshgrid(sy, sx, indexing="ij")
            if TORCH_1_10
            else torch.meshgrid(sy, sx)
        )
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def _dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class _DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        x = x.view(b, 4, self.c1, a)
        x = x.transpose(2, 1)
        x = x.softmax(1)
        x = self.conv(x)
        x = x.view(b, 4, a)
        return x


class BoxDecoderPytorchV1:
    def __init__(self, stride, conf_thres) -> None:
        # self.no = self.nc + self.reg_max * 4
        self.conf_thres = conf_thres
        self.reg_max = 16
        self.stride = stride
        self.shape = None
        self.anchors = None
        self.strides = None

        self.dfl = _DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def _decode_box(self, feats_box, feats_cls):
        x = [t for tensors in zip(feats_box, feats_cls) for t in tensors]
        # print(len(x))
        x = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in x]

        shape = x[0].shape

        # concat box and conf
        # sigmoid already included in model
        nc = x[1].shape[1]
        no = nc + self.reg_max * 4

        x = [torch.cat((x[2 * i], x[2 * i + 1]), 1) for i in range(len(x) // 2)]

        if self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in _make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in x], 2)

        box, cls = x_cat.split((self.reg_max * 4, nc), 1)
        dbox = (
            _dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=False, dim=1)
            * self.strides
        )
        y = torch.cat((dbox, cls.sigmoid()), 1)

        return y, nc

    def _filter_conf(self, y, nc, as_np=True):
        out = []

        for x in y:
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose(1, 0)
            box, cls, extra = x[:, :4], x[:, 4 : 4 + nc], x[:, 4 + nc :]
            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = cls.max(1, keepdim=True)
            conf_mask = conf.view(-1) > self.conf_thres
            x = torch.cat((box, conf, j.float(), extra), 1)[conf_mask]
            if as_np:
                x = x.numpy()
            out.append(x)

        return out

    def __call__(self, feats_box, feats_cls):
        return self._filter_conf(*self._decode_box(feats_box, feats_cls))


BoxDecoderPytorch = BoxDecoderPytorchV1
