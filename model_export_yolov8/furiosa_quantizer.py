import argparse
import os
import random
from tqdm import tqdm

import cv2
import onnx
from onnx.utils import Extractor
import numpy as np

from furiosa.optimizer import optimize_model
from furiosa.quantizer import (
    get_pure_input_names,
    get_output_names,
    quantize,
    Calibrator,
    CalibrationMethod,
    ModelEditor,
    TensorType,
)

INPUT_SHAPE = (640, 640)


def main():
    args = build_argument_parser()

    onnx_path = args.onnx_path
    output_path = args.output_path
    calib_data = args.calib_data
    calib_count = args.calib_count
    input_name = args.model_input_name

    f32_onnx_model = onnx.load_model(onnx_path)
    extracted_onnx_model = extract_model(f32_onnx_model, input_name)
    optimized_onnx_model = optimize_model(
        model=extracted_onnx_model,
        opset_version=13,
        input_shapes={input_name: [1, 3, *INPUT_SHAPE]},
    )

    calib_data_names = os.listdir(calib_data)
    calib_data_names = random.choices(calib_data_names, k=calib_count)

    calibrator = Calibrator(optimized_onnx_model, CalibrationMethod.MIN_MAX_ASYM)
    for image_name in tqdm(
        calib_data_names, desc="Calibration", unit="image", mininterval=0.5
    ):
        image_path = os.path.join(calib_data, image_name)
        image, _ = preprocess(image_path, new_shape=INPUT_SHAPE)
        calibrator.collect_data([[image]])

    ranges = calibrator.compute_range()

    editor = ModelEditor(optimized_onnx_model)
    input_name1 = get_pure_input_names(optimized_onnx_model)[0]
    editor.convert_input_type(input_name1, TensorType.UINT8)

    i8_onnx_model = quantize(optimized_onnx_model, ranges)

    with open(output_path, "wb") as f:
        f.write(bytes(i8_onnx_model))

    print(f"Completed quantinization >> {output_path}")


def preprocess(img_path, new_shape=(640, 640)):
    img = cv2.imread(img_path)

    img, preproc_params = letterbox(img, new_shape, auto=False)

    img = img.transpose((2, 0, 1))[::-1]
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
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


def extract_model(model: onnx.ModelProto, input_name):
    """Cut off the post-processing components."""
    input_to_shape = [(input_name, (1, 3, *INPUT_SHAPE))]

    output_to_shape = (
        (
            "/model.22/cv2.0/cv2.0.2/Conv_output_0",
            (1, 64, int(INPUT_SHAPE[0] / 8), int(INPUT_SHAPE[1] / 8)),
        ),
        (
            "/model.22/cv3.0/cv3.0.2/Conv_output_0",
            (1, 80, int(INPUT_SHAPE[0] / 8), int(INPUT_SHAPE[1] / 8)),
        ),
        (
            "/model.22/cv2.1/cv2.1.2/Conv_output_0",
            (1, 64, int(INPUT_SHAPE[0] / 16), int(INPUT_SHAPE[1] / 16)),
        ),
        (
            "/model.22/cv3.1/cv3.1.2/Conv_output_0",
            (1, 80, int(INPUT_SHAPE[0] / 16), int(INPUT_SHAPE[1] / 16)),
        ),
        (
            "/model.22/cv2.2/cv2.2.2/Conv_output_0",
            (1, 64, int(INPUT_SHAPE[0] / 32), int(INPUT_SHAPE[1] / 32)),
        ),
        (
            "/model.22/cv3.2/cv3.2.2/Conv_output_0",
            (1, 80, int(INPUT_SHAPE[0] / 32), int(INPUT_SHAPE[1] / 32)),
        ),
    )

    input_to_shape = {
        tensor_name: [
            onnx.TensorShapeProto.Dimension(dim_value=dimension_size)
            for dimension_size in shape
        ]
        for tensor_name, shape in input_to_shape
    }
    output_to_shape = {
        tensor_name: [
            onnx.TensorShapeProto.Dimension(dim_value=dimension_size)
            for dimension_size in shape
        ]
        for tensor_name, shape in output_to_shape
    }

    extracted_model = Extractor(model).extract_model(
        input_names=list(input_to_shape), output_names=list(output_to_shape)
    )

    for value_info in extracted_model.graph.input:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(input_to_shape[value_info.name])
    for value_info in extracted_model.graph.output:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(output_to_shape[value_info.name])

    return extracted_model


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_path", type=str, default="yolov8n.onnx", help="Path to onnx file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="quantized_model.onnx",
        help="Path to i8 onnx file",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="../coco/val2017",
        help="Path to calibration data containing image files",
    )
    parser.add_argument(
        "--calib_count",
        default=100,
        type=int,
        help="How many images to use for calibration",
    )
    parser.add_argument(
        "--model_input_name", type=str, default="images", help="the model's input name"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
