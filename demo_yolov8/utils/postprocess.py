import numpy as np
import torch
import torchvision
import cv2
from utils.info import *
import time
import pkg_resources as pkg
from torch import nn


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


def _compute_stride():
    img_h = 640
    # output_shapes = self.output_shapes[::2]
    feat_h = np.float32([shape[0] for shape in [(80, 80), (40, 40), (20, 20)]])
    strides = img_h / feat_h
    return strides


STRIDE = _compute_stride()


def postproc(feats, conf_thres=0.35, iou_thres=0.65):
    boxes_batched = []
    # scale, (padw, padh) = preproc_params
    box_decoder = BoxDecoderPytorch(stride=_compute_stride(), conf_thres=conf_thres)
    # if isinstance(scale, (tuple, list)):
    #    assert len(scale) == 2 and scale[0] == scale[1]
    #    scale = scale[0]

    for i in range(feats[0].shape[0]):
        feats = [f[i : i + 1] for f in feats]
        feats_box, feats_cls = feats[0::2], feats[1::2]
        boxes_dec = box_decoder(feats_box, feats_cls)
        boxes = non_max_suppression(boxes_dec, iou_thres=iou_thres)[0]
        # boxes[:, [0, 2]] = (1 / scale) * (boxes[:, [0, 2]] - padw)
        # boxes[:, [1, 3]] = (1 / scale) * (boxes[:, [1, 3]] - padh)
        boxes_batched.append(boxes)

    return boxes_batched


def non_max_suppression(prediction, iou_thres=0.45, class_agnostic=True):
    # Checks
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height

    output = []
    for x in prediction:  # image index, image inference
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Batched NMS
        if not class_agnostic:
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        else:
            boxes, scores = x[:, :4], x[:, 4]

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        output.append(x[i].numpy())

    return output


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img

    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img


def draw_bbox(img, bbox, preproc_param):
    # img = cv2.imread(img_path)
    ratio, dwdh = preproc_param

    bbox[:, [0, 2]] = (1 / ratio) * (bbox[:, [0, 2]] - dwdh[0])
    bbox[:, [1, 3]] = (1 / ratio) * (bbox[:, [1, 3]] - dwdh[1])

    for i, box in enumerate(bbox):
        x0, y0, x1, y1 = [int(i) for i in box[:4]]
        mbox = np.array([x0, y0, x1, y1])
        mbox = mbox.round().astype(np.int32).tolist()
        score = box[4]
        class_id = int(box[5])

        color = COLORS_10[class_id % len(COLORS_10)]
        label = f"{CLASSES[class_id]} ({score:.2f})"

        img = plot_one_box([x0, y0, x1, y1], img, color, label)

    return img
