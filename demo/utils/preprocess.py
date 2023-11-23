import cv2
import numpy as np


def preproc(img: str, new_shape=(640, 640)):

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    img, preproc_params = letterbox(img, new_shape, auto=False)

    img = img.transpose((2, 0, 1))[::-1]
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return img, preproc_params


def letterbox(
    img, new_shape, color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    h, w = img.shape[:2]

    ratio = min(new_shape[0] / h, new_shape[1] / w)

    if not scaleup:
        ratio = min(ratio, 1.0)

    new_unpad = int(round(ratio * w)), int(round(ratio * h))
    dw, dh = (new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1])

    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        interpolation = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        img = cv2.resize(img, new_unpad, interpolation=interpolation)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, (ratio, (dw, dh))
