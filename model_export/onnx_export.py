import argparse
import os
import sys
from pathlib import Path

import torch
import onnx
from onnx import shape_inference

root_path = Path(__file__).parent
INPUT_SHAPE = (640, 640)


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov7.pt",
        help="Path to weight file (.pt or .pth)",
    )  # --weights
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="yolov7.onnx",
        help="Path of onnx file to generate",
    )  # --onnx_path
    parser.add_argument(
        "--model_input_name", type=str, default="images", help="the model's input name"
    )
    parser.add_argument(
        "--model_output_name",
        type=str,
        default="output",
        help="the model's output name",
    )
    parser.add_argument(
        "--custom", type=str, default="no", help="custom model or not (yes / no)"
    )
    args = parser.parse_args()
    return args


def main():
    args = build_argument_parser()

    weights = args.weights
    onnx_path = args.onnx_path
    input_names = args.model_input_name
    output_names = args.model_output_name
    custom = True if args.custom == "yes" else False

    device = torch.device("cpu")
    model = torch.hub.load("WongKinYiu/yolov7", "custom", weights)

    x = torch.zeros(1, 3, *INPUT_SHAPE).to(device)
    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    torch.onnx.export(
        model,
        x,
        onnx_path,
        opset_version=13,
        input_names=[input_names],
        output_names=[output_names],
    )

    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)


if __name__ == "__main__":
    main()
