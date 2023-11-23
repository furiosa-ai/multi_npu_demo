import numpy as np
import torch
import torchvision
import cv2
from utils.info import *


def postproc(prediction, conf_thres, iou_thres):
    outputs = []
    for i in range(NUM_LAYERS):
        batch_size, _, ny, nx = prediction[i].shape
        prediction[i] = np.transpose(
            np.reshape(
                prediction[i], (batch_size, NUM_ANCHORS, NUM_CLASSES + 5, ny, nx)
            ),
            axes=(0, 1, 3, 4, 2),
        )

        xy, wh, conf = np.split(sigmoid(prediction[i]), [2, 4], axis=4)
        xy = (xy * 2 + GRID[i]) * STRIDE[i]
        wh = (wh * 2) ** 2 * ANCHOR_GRID[i]
        y = np.concatenate((xy, wh, conf), axis=4)
        outputs.append(np.reshape(y, (1, NUM_ANCHORS * nx * ny, NUM_CLASSES + 5)))

    outputs = np.concatenate(outputs, axis=1)
    prediction = non_max_suppression(outputs, conf_thres, iou_thres)
    return prediction


def sigmoid(x: np.ndarray) -> np.ndarray:
    # pylint: disable=invalid-name
    return 1 / (1 + np.exp(-x))


# https://github.com/ultralytics/yolov5/blob/v7.0/utils/general.py#L884-L999
def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    agnostic: bool = False,
    max_det: int = 300,
):
    # pylint: disable=invalid-name,too-many-locals

    batch_size = prediction.shape[0]
    candidates = prediction[..., 4] > conf_thres

    assert 0 <= conf_thres <= 1, conf_thres
    assert 0 <= iou_thres <= 1, iou_thres

    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000

    output = [np.zeros((0, 6))] * batch_size
    for xi, x in enumerate(prediction):
        x = x[candidates[xi]]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        box = xywh2xyxy(x[:, :4])

        i, j = np.where(x[:, 5:] > conf_thres)
        x = np.concatenate(
            (
                box[i],
                x[i, j + 5, np.newaxis].astype(np.float32),
                j[:, np.newaxis].astype(np.float32),
            ),
            axis=1,
        )

        n = x.shape[0]
        if not n:
            continue

        if n > max_nms:
            x = x[np.argsort(x[:, 4])[::-1][:max_nms]]

        classes = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + classes, x[:, 4]  # boxes (offset by class), scores

        i = torchvision.ops.nms(
            torch.from_numpy(boxes), torch.from_numpy(scores), iou_thres
        ).numpy()
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]

    return output


# https://github.com/ultralytics/yolov5/blob/v7.0/utils/general.py#L760-L767
def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    # pylint: disable=invalid-name
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def init_grid():
    grid = [np.zeros(1)] * NUM_LAYERS
    anchor_grid = [np.zeros(1)] * NUM_LAYERS

    for i, (ny, nx) in zip(  #  pylint: disable=invalid-name
        range(NUM_LAYERS), ((80, 80), (40, 40), (20, 20))
    ):
        grid[i], anchor_grid[i] = make_grid(nx, ny, i)

    return grid, anchor_grid


def make_grid(nx: int, ny: int, i: int):
    shape = 1, NUM_ANCHORS, ny, nx, 2
    y, x = np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32)
    yv, xv = np.meshgrid(y, x, indexing="ij")
    grid = np.broadcast_to(np.stack((xv, yv), axis=2), shape) - 0.5
    anchor_grid = np.broadcast_to(
        np.reshape(ANCHORS[i] * STRIDE[i], (1, NUM_ANCHORS, 1, 1, 2)), shape
    )
    return grid, anchor_grid


GRID, ANCHOR_GRID = init_grid()


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
